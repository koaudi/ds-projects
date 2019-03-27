#This script imports rwanda raw data, cleans & tags recurring customers
#then we split into training vs testing dataset and create a classification model using logistic regression to predict which customers will be repeat customers
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from pylab import rcParams
from settings import MAIN_DIRECTORY

os.chdir(MAIN_DIRECTORY)
os.getcwd()

# Loading the CSV with pandas
rw_df = pd.read_csv('csv/rw/rw_customers_products.csv', low_memory = False, error_bad_lines = False)

#PREPPING DATA - mark repeat customers and keep relevant variables for feature generation
rw_df = rw_df[rw_df.agent_order == False]
rw_df.sort_values(by = 'order_id', inplace = True)
order_count = pd.DataFrame(rw_df.groupby('customer_id')['order_id'].nunique()).reset_index() 
order_count['repeat_customer'] = np.where(order_count.order_id >= 2, 1,0)
data = rw_df[['customer_id','order_source','payment_method','gender','device']]
data.drop_duplicates(subset='customer_id', keep='first', inplace=True)
data = pd.merge(data,order_count[['customer_id','repeat_customer']], on='customer_id', how = 'left')
data.repeat_customer.fillna(0,inplace = True)
data["repeat_customer"] = data["repeat_customer"].astype(int)
data.drop(['customer_id'], axis=1, inplace=True)

#Check values
sizes = data['repeat_customer'].value_counts(sort = True)
colors = ["blue",'green'] 
rcParams['figure.figsize'] = 5,5

# Plot Repeat customers
plt.pie(sizes, labels = sizes.index, colors=colors,autopct='%1.1f%%', shadow=True, startangle=270,)
plt.title('Percentage of Repeat Customers')
#plt.show()

clean_data = pd.get_dummies(data,columns = ['order_source','payment_method','gender','device'])

y = clean_data['repeat_customer'].values
x = clean_data.drop(labels = ["repeat_customer"],axis = 1)

# Create Train & Test Data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

#Fit log regression to training data
#Step 1. Let’s Import the model we want to use from sci-kit learn
#Step 2. We make an instance of the Model
#Step 3. Is training the model on the training data set and storing the information learned from the data
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

# Use score method to get accuracy of model
score = model.score(x_test, y_test)
print(score)

###############
data.groupby(["gender", "repeat_customer"]).size().unstack().plot(kind='bar', stacked=True, figsize=(30,10)) 
plt.show()