#https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

os.chdir('C:/Users/KOFA-X1/Desktop/github/ds-learnings/predicting_churn')
os.getcwd()

# Loading the CSV with pandas
clean_data = pd.read_csv('training_data.csv', low_memory = False, error_bad_lines = False)

y = clean_data['repeat_customer'].values
x = clean_data.drop(labels = ["repeat_customer"],axis = 1)

#Fit log regression to training data
#Step 1. Let’s Import the model we want to use from sci-kit learn
#Step 2. We make an instance of the Model
#Step 3. Is training the model on the training data set and storing the information learned from the data
from sklearn.linear_model import LogisticRegression
regression = LogisticRegression()
regression.fit(x, y)


#Make pickle file of model
pickle.dump(regression, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1, 0, 0]]))
