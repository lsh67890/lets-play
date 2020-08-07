"""Auction House

This programme allows the user to input auction actions and get the final output
for each item in the auction.
"""


class Item:
    """Class for items in the auction.

    Attributes:
        start_time (int): auction start time
        timestamp (int): time for the bid
        item (str): unique code for the item
        reserve_price (float): initial reserve price
        close_time (int): auction closing time
        bid_count (int): number of valid bids
        user_id (int): id of the user
        status (str): status indicating if it's sold or unsold
        bids (list): list of prices of the bids
    """
    
    def __init__(self, start_time, item, reserve_price, close_time):
        """Constructor for Item class."""
        
        self.start_time = start_time
        self.timestamp = start_time
        self.item = item
        self.reserve_price = reserve_price
        self.close_time = close_time
        self.bid_count = 0
        self.user_id = ''
        self.status = 'UNSOLD'
        self.bids = []


    def is_valid_bid(self, timestamp, bid_amount):
        """Function to check if a bid is valid.

        Args:
            timestamp (int): time of bidding
            bid_amount (float): bidding price

        Returns:
            bool: whether it's a valid bid or not
        """
        
        if self.start_time < timestamp <= self.close_time:
            if not self.bids:
                self.bid_count += 1
                self.bids.append(bid_amount)
                return True
            if bid_amount > self.bids[-1]:
                self.bid_count += 1
                return True
        else:
            return False
        
        
    def bid(self, bid_amount, user_id, timestamp):
        """Function to put in a bid.

        Args:
            bid_amount (float): bidding price
            user_id (int): id of the bidder
            timestamp (int): time of bidding
        """
        
        valid_bid = self.is_valid_bid(timestamp, bid_amount)
        if valid_bid:
            if bid_amount > max(self.bids):
                self.user_id = user_id
                self.timestamp = timestamp
                self.bids.append(bid_amount)


    def get_price_paid(self):
        """Function to get the final price paid.

        Returns:
            str: final price paid
        """
        if self.status == 'SOLD':
            if len(self.bids)==1:
                # if there is only one valid bid,
                # then the final price is the reserve price.
                return str(self.reserve_price)
            else:
                # otherwise the second highest valid bid.
                return str(self.bids[-2])


    def update_status(self):
        if not self.bids:
            pass
        elif self.bids[-1] > self.reserve_price:
            self.status = 'SOLD'
        else:
            self.user_id = ''


def split_args(input_str):
    """Function to split the input string into arguments.

    Args:
        input_str (str): an act in the auction

    Returns:
        act_args (list): list of argument for the act
    """
    
    act_args = input_str.split(',')
    return act_args
    

def make_output(items):
    """Function to generate final output.

    Args:
        items (dict): a dictionary containing items

    Returns:
        output (list): list of final output strings
    """
    
    output = []
    for i in items:
        items[i].bids.sort()
        items[i].update_status()
        if not items[i].bids:
            highest = lowest = ''
        else:
            highest = str(max(items[i].bids))
            lowest = str(min(items[i].bids))
        if items[i].status == 'SOLD':
            price_paid = items[i].get_price_paid()
        else:
            price_paid = str(0.00)

        final = ('Item: ' + items[i].item + ' ' + items[i].status
                 + ' For user id:' + str(items[i].user_id)
                 + ' Price paid:' + price_paid
                 + ' Closed at:' + str(items[i].close_time)
                 + ' Highest bid:' + highest
                 + ' Lowest bid:' + lowest
                 )
        output.append(final)
    return output


def add_selling_item(timestamp, item, reserve_price, close_time):
    """Function to add item to sell on the auction.

    Args:
        timestamp (int): start time of auction
        item (str): unique code of the item
        reserve_price (float): reserved price of the item
        close_time (int): closing time of the auction

    Returns:
        new_item (obj): added selling item
    """
    
    if timestamp > close_time:
        print('Starting time on {0} is later than closing time!'.format(item))
    new_item = Item(
        timestamp, item, reserve_price, close_time
    )
    return new_item

    
def main():
    """Main function to run the auction house"""
    items = {}

    # initialise heartbeat and closing time
    heartbeat = 0
    close = 24
    closings = []
    print('Choose your action and provide details in the following format: \n',
        '*For selling (do not change SELL): \n',
        '  start_time,user_id,SELL,item,reserve_price,close_time \n',
        '  Example: 12,2,SELL,lamp_1,15.00,20 \n',
        '*For bidding (do not change BID): \n',
        '  bidding_time,user_id,BID,item,bid_amount \n'
        '   Example: 14,5,BID,lamp_1,15.5'
        )
    while heartbeat <= close:
        choice = input()
        act_args = split_args(choice)
        try:
            timestamp = int(act_args[0])
            user_id = int(act_args[1])
            item = act_args[3]
        except ValueError:
            print(choice, 'is not a valid value!')
            break
        if 'SELL' in act_args:
            reserve_price = float(act_args[4])
            close_time = int(act_args[5])
            new_item = add_selling_item(timestamp, item, reserve_price, close_time)
            items[item] = new_item
            closings.append(close_time)
            close = max(closings)
        elif 'BID' in act_args:
            bid_amount = float(act_args[4])
            try:
                items[item].bid(bid_amount, user_id, timestamp)
                if items[item].bid_count == 1:
                    items[item].user_id = user_id
            except KeyError:
                print('{0} is not on auction!'.format(item))
     
        heartbeat = timestamp

    result = make_output(items)
    print('+++AUCTION RESULT+++')
    for res in result:
        print(res)


if __name__=='__main__':
    main()
