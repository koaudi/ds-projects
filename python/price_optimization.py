#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt
from settings import MAIN_DIRECTORY

os.chdir(MAIN_DIRECTORY)
os.getcwd()
usd_rwf = 860

rw_df = pd.read_csv('csv/rw/rw_customers_products.csv',low_memory = False, error_bad_lines = False)

rw_df['product_sales_total'] = rw_df.product_sales_total / usd_rwf 

price = rw_df[['product_slug','quantity','product_sales_total']]
price = price[price.quantity > 0]
price['avg_sales_price'] = price.product_sales_total / price.quantity
avg = pd.DataFrame(price.groupby('product_slug')['avg_sales_price'].mean()).reset_index()
avg = avg[avg.avg_sales_price > 0 ]

#NORMALIZING THE PRICE USING VARIOUS TECHNIQUES
avg['log_price'] = avg.avg_sales_price.apply(np.log) + 1
avg['sqrt_price'] = np.sqrt(avg.avg_sales_price)
avg['cube_price'] = avg.avg_sales_price**(1./3.)


from sklearn import preprocessing
x_array = np.array(avg.avg_sales_price)
zscore_price = preprocessing.normalize([x_array])




sns.set(color_codes=True)
#Plotting univariate distributions



sns.distplot(avg.avg_sales_price.astype(int))
plt.show()

sns.distplot(avg.log_price.astype(int))
plt.show()

sns.distplot(avg.sqrt_price.astype(int))
plt.show()

sns.distplot(avg.cube_price.astype(int))
plt.show()

sns.distplot(zscore_price)
plt.show()