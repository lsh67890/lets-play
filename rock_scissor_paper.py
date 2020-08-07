# Rock scissor paper game: User vs Computer
# Based on "Best of 3" rule

import random

CHOICES = {1: 'rock', 2: 'scissor', 3:'paper'}
NUMBERS = [1, 2, 3]

def get_result(your_choice, comp_choice):
    unchosen = [e for e in NUMBERS if e not in (your_choice, comp_choice)][0]
    result = your_choice - comp_choice
    another_result = your_choice - unchosen

    if your_choice == comp_choice:
        print('Draw!')
        return None # draw

    final = None
    if (your_choice % 2) == 0:
        if result < another_result:
            final = 1 # win
        else:
            final = -1 # lose
    elif result > another_result:
        final = 1 # win
    else:
        final = -1 # lose
    return final


def choose_winner(your_score, comp_score):
    if your_score > comp_score:
        print('----The winner is...YOU!----')
    else:
        print('----The winner is...COMPUTER!----')

def main():
    total=your_score=comp_score=0

    while total < 3:
        print('Enter your choice: 1=rock, 2=scissor, 3=paper')
        your_choice = int(input())
        comp_choice = random.choice(NUMBERS)
        print('You:', CHOICES[your_choice], 'Computer:', CHOICES[comp_choice])

        result = get_result(your_choice, comp_choice)
        if result == 1:
            your_score = your_score + 1
        if result == -1:
            comp_score = comp_score + 1
        print('Current score:', 'You', your_score, 'Computer', comp_score)
        if your_score == 2 or comp_score == 2:
            break
        total = total + 1
        if total == 3 and your_score == comp_score:
            print('***Three rounds done yet the same score! Resetting game...***')
            total=your_score=comp_score=0

    choose_winner(your_score, comp_score)


if __name__=='__main__':
    main()
