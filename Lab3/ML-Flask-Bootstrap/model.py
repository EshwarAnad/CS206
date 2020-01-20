# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')  #Reading the Dataset

dataset['experience'].fillna(0, inplace=True)   #Filling all the NaN values of Experience Column of Dataset with 0 experience.

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True) #Filling all NaN values of Test-Score Column with the mean Values of test_score Column

X = dataset.iloc[:, :3]   #Selecting All the three Columns of the dataset.

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]  

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]  #Gives the Salary Column

#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving and Loading can be done in 2 ways in simple Regression, one of them is pickle(pkl) and Other one is joblib library

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))