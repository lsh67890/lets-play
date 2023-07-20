import os
os.environ['JAVA_HOME'] = "C:/Program Files/Java/jdk-20/bin/server"

import random
import numpy as np
import tensorflow as tf
from konlpy.tag import Okt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Lambda, Layer, Embedding, LayerNormalization

EPOCHS = 200
NUM_WORDS = 2000

# DOT-SCALED ATTENTION
class DotScaledAttention(Layer):
    def __init__(self, d_emb, d_reduced, masked=False): # embedding dimension
        super().__init__()
        self.q = Dense(d_reduced, input_shape=(-1, d_emb))
        self.k = Dense(d_reduced, input_shape=(-1, d_emb))
        self.v = Dense(d_reduced, input_shape=(-1, d_emb))
        self.scale = Lambda(lambda x: x/np.sqrt(d_reduced))
        self.masked = masked

    def call(self, x, training=None, mask=None): # (q, k, v)
        q = self.scale(self.q(x[0]))
        k = self.k(x[1])
        v = self.v(x[2])

        k_T = tf.transpose(k, perm=[0, 2, 1])
        comp = tf.matmul(q, k_T)

        if self.masked:
            length = tf.shape(comp)[-1]
            mask = tf.fill((length, length), -np.inf)
            mask = tf.linalg.band_part(mask, 0, -1) # get upper triangle
            mask = tf.linalg.set_diag(mask, tf.zeros((length))) # set diagonal to zeros to avoid operations with infinity
            comp += mask
        comp = tf.nn.softmax(comp, axis=-1)
        return tf.matmul(comp, v)

# MULTI-HEAD ATTENTION
class MultiHeadAttention(Layer):
    def __init__(self, num_head, d_emb, d_reduced, masked=False):
        super().__init__()
        self.attention_list = list()
        for _ in range(num_head):
            self.attention_list.append(DotScaledAttention(d_emb, d_reduced, masked))
        self.linear = Dense(d_emb, input_shape=(-1, num_head * d_reduced))

    def call(self, x, training=None, mask=None):
        attention_list = [a(x) for a in self.attention_list]
        concat = tf.concat(attention_list, axis=-1)
        return self.linear(concat)

# ENCODER
class Encoder(Layer):
    def __init__(self, num_head, d_reduced):
        super().__init__()
        self.num_head = num_head
        self.d_r = d_reduced

    def build(self, input_shape):
        self.multi_attention = MultiHeadAttention(self.num_head, input_shape[-1], self.d_r)
        self.layer_norm1 = LayerNormalization(input_shape=input_shape)
        self.dense1 = Dense(input_shape[-1] * 4, input_shape=input_shape, activation='relu')
        self.dense2 = Dense(input_shape[-1], input_shape=self.dense1.compute_output_shape(input_shape))
        self.layer_norm2 = LayerNormalization(input_shape=input_shape)
        super().build(input_shape)

    def call(self, x, training=None, mask=None):
        h = self.multi_attention((x, x, x))
        ln1 = self.layer_norm1(x + h)
        h = self.dense2(self.dense1(ln1))
        return self.layer_norm2(h + ln1)

    def compute_output_shape(self, input_shape):
        return input_shape

# DECODER
class Decoder(Layer):
    def __init__(self, num_head, d_reduced):
        super().__init__()
        self.num_head = num_head
        self.d_r = d_reduced

    def build(self, input_shape):
        self.self_attention = MultiHeadAttention(self.num_head, input_shape[0][-1], self.d_r, masked=True)
        self.layer_norm1 = LayerNormalization(input_shape=input_shape)
        self.multi_attention = MultiHeadAttention(self.num_head, input_shape[0][-1], self.d_r)
        self.layer_norm2 = LayerNormalization(input_shape=input_shape)
        self.dense1 = Dense(input_shape[0][-1] * 4, input_shape=input_shape[0], activation='relu')
        self.dense2 = Dense(input_shape[0][-1], input_shape=self.dense1.compute_output_shape(input_shape[0]))
        self.layer_norm3 = LayerNormalization(input_shape=input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None): # (x, context)
        x, context = inputs
        h = self.self_attention((x, x, x))
        ln1 = self.layer_norm1(x + h)

        h = self.multi_attention((ln1, context, context))
        ln2 = self.layer_norm2(ln1 + h)

        h = self.dense2(self.dense1(ln2))
        return self.layer_norm3(h + ln2)

    def compute_output_shape(self, input_shape):
        return input_shape

# POSITIONAL ENCODING
class PositionalEncoding(Layer):
    def __init__(self, max_len, d_emb):
        super().__init__()
        self.sinusoidal_encoding = np.array([self.get_positional_angle(pos, d_emb) for pos in range(max_len)], dtype=np.float32)
        self.sinusoidal_encoding[:, 0::2] = np.sin(self.sinusoidal_encoding[:, 0::2])
        self.sinusoidal_encoding[:, 1::2] = np.cos(self.sinusoidal_encoding[:, 1::2])
        self.sinusoidal_encoding = tf.cast(self.sinusoidal_encoding, dtype=tf.float32)

    def call(self, x, training=None, mask=None):
        return x + self.sinusoidal_encoding[:tf.shape(x)[1]]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_angle(self, pos, dim, d_emb):
        return pos / np.power(10000, 2 * (dim // 2) / d_emb)

    def get_positional_angle(self, pos, d_emb):
        return [self.get_angle(pos, dim, d_emb) for dim in range(d_emb)]

# TRANSFORMER ARCHITECTURE
class Transformer(Model):
    def __init__(self, src_vocab, dst_vocab, max_len, d_emb, d_reduced, n_enc_layer, n_dec_layer, num_head):
        super().__init__()
        self.enc_emb = Embedding(src_vocab, d_emb)
        self.dec_emb = Embedding(dst_vocab, d_emb)
        self.pos_enc = PositionalEncoding(max_len, d_emb)

        self.encoder = [Encoder(num_head, d_reduced) for _ in range(n_enc_layer)]
        self.decoder = [Decoder(num_head, d_reduced) for _ in range(n_dec_layer)]
        self.dense = Dense(dst_vocab, input_shape=(-1, d_emb))

    def call(self, inputs, training=None, mask=None):
        src_sent, dst_sent_shifted = inputs

        h_enc = self.pos_enc(self.enc_emb(src_sent))
        for enc in self.encoder:
            h_enc = enc(h_enc)

        h_dec = self.pos_enc(self.dec_emb(dst_sent_shifted))
        for dec in self.decoder:
            h_dec = dec([h_dec, h_enc])

        return tf.nn.softmax(self.dense(h_dec), axis=-1)

# Dataset preparation
dataset_file = "chatbot_data.csv"
okt = Okt()

with open(dataset_file, encoding='utf-8') as file:
    lines = file.readlines()
    seq = [' '.join(okt.morphs(line)) for line in lines]

questions = seq[::2]
answers = ['\t' + lines for lines in seq[1::2]]

num_sample = len(questions)

perm = list(range(num_sample))
random.seed(0)
random.shuffle(perm)

train_q = list()
train_a = list()
test_q = list()
test_a = list()

for idx, qna in enumerate(zip(questions, answers)):
    q, a = qna
    if perm[idx] > num_sample // 5:
        train_q.append(q)
        train_a.append(a)
    else:
        test_q.append(q)
        test_a.append(a)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
tokenizer.fit_on_texts(train_q + train_a)

train_q_seq = tokenizer.texts_to_sequences(train_q)
train_a_seq = tokenizer.texts_to_sequences(train_a)
test_q_seq = tokenizer.texts_to_sequences(test_q)
test_a_seq = tokenizer.texts_to_sequences(test_a)

x_train = tf.keras.preprocessing.sequence.pad_sequences(train_q_seq, value=0, padding='pre', maxlen=64)
y_train = tf.keras.preprocessing.sequence.pad_sequences(train_a_seq, value=0, padding='post', maxlen=65)
y_train_shifted = np.concatenate([np.zeros((y_train.shape[0], 1)), y_train[:, 1:]], axis=1)

x_test = tf.keras.preprocessing.sequence.pad_sequences(test_q_seq, value=0, padding='pre', maxlen=64)
y_test = tf.keras.preprocessing.sequence.pad_sequences(test_a_seq, value=0, padding='post', maxlen=65)

# train using keras
transformer = Transformer(NUM_WORDS, NUM_WORDS, 128, 16, 16, 2, 2, 4) # initialise new transformer model
transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
transformer.fit([x_train, y_train_shifted], y_train, batch_size=5, epochs=EPOCHS)

'''
Epoch 1/200
80/80 [==============================] - 19s 56ms/step - loss: 6.1116 - accuracy: 0.8482
Epoch 200/200
80/80 [==============================] - 4s 54ms/step - loss: 0.0013 - accuracy: 0.9995
'''
