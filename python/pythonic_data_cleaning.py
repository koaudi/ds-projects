import pandas as pd
import numpy


#UPDATING PRODUCT QUANTITIES
product_dict = {17729:2,30926:3,19056:3,19056:3,75425:2,162587:3}
def change_quantity(data):
    for key,value in product_dict.items():
        data.loc[data.product_id == key, 'quantity'] = data.quantity * value
    return data

