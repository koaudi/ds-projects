#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import glob
from settings import MAIN_DIRECTORY

os.chdir(MAIN_DIRECTORY)
os.getcwd()

rw_df = pd.read_csv('csv/rw/rw_customers_products.csv',low_memory = False, error_bad_lines = False)

price = rw_df[['product_slug','quantity','product_sales_total']]
price = price[price.quantity > 0]
price['avg_sales_price'] = price.product_sales_total / price.quantity
avg = pd.DataFrame(price.groupby('product_slug')['avg_sales_price'].mean()).reset_index()
avg = avg[avg.avg_sales_price > 0 ]
avg['log_price'] = avg.avg_sales_price.apply(np.log)

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set(color_codes=True)
#Plotting univariate distributions



avg.sort_values('log_price')

sns.distplot(avg.avg_sales_price.astype(int))
plt.show()

sns.distplot(avg.log_price.astype(int))
plt.show()
