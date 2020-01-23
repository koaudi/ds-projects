import os
import pandas as pd
import numpy as np
import re
from settings import MAIN_DIRECTORY

os.chdir(MAIN_DIRECTORY)
os.getcwd()
#test

products = pd.read_csv('csv/ke_products.csv',low_memory = False, error_bad_lines = False,encoding='latin-1')
products['Categories'] = products['Categories'].astype('str')
for rows_in_column in products[['Categories']]:
    columnseriesbj = products[rows_in_column]
    message = columnseriesbj.values

x = re.search(r"Brands\s>\s\w+\,",message)
print(x.group())


products['brands'] = products[products.Categories.str.extract(r"1")]


for num in range(1,51):
    if num %3 == 0 and num %5 == 0: 
        print('fizzbuzz')
        continue
    elif num % 3 == 0 : 
        print("fizz")
        continue
    elif num % 5 == 0 : 
        print("buzz")
        continue
    print(num)
