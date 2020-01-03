import pandas as pd
import numpy


#UPDATING PRODUCT QUANTITIES
product_dict = {17729:2,30926:3,19056:3,19056:3,75425:2,162587:3}
def change_quantity(data):
    for key,value in product_dict.items():
        data.loc[data.product_id == key, 'quantity'] = data.quantity * value
    return data

#FIX DATES
df['date'] = df['paid_date'].dt.date
df['paid_date'].fillna(df.order_date, inplace=True)
df['order_date'] = pd.to_datetime(df.order_date)
df['year'] = df.paid_date.dt.to_period('A')
df['year_month_1'] = df.order_date.dt.to_period('M')
df['week'] = df['paid_date'].dt.to_period('W')
df['week'] = df.week.astype(str)
df['weekstart'], df['weekend'] = df['week'].str.split('/', 1).str
