# Importing necessary classes that I will use
import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Reading movie_dataset.csv file
movie = pd.read_csv('movie_dataset.csv')

# Displaying first five rows of my data
print (movie.head())

# Displaying number of rows and columns of the DataFrame
print ('\n\nNumber of rows and columns: ', movie.shape)


cv = CountVectorizer(max_features=2500)

# Vectorizing words and storing in variable X(predictor)
X = cv.fit_transform(movie['text']).toarray()

# Our target or output 
y = movie.iloc[:,-1].values


# Spliting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)


# Fitting and predicting Gaussian
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)


# Print the Gaussian accuracy score
print("\nGaussian Naive Bayes Model Accuracy   : ", accuracy_score(y_test, y_pred_gnb))

# Printing the Gaussian confusion matrices
cmGN=confusion_matrix(y_test, y_pred_gnb)
print ('Gaussian Confusion Matrix: \n', cmGN)

#----------------------------------------
# Training and predicting dataset using neural network
nn = MLPClassifier(activation = 'logistic', solver='sgd', hidden_layer_sizes=(5,100), random_state=3)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)

# Printing the Neural Network accuracy score
print('\n\n\nNeural Network Model Accuracy : ', accuracy_score(y_test, nn_pred))

# Printing the Neural Network confusion matrix
cunfuseNeurals=confusion_matrix(y_test, nn_pred)
print ('Neural Network Confusion Matrix:\n ', cunfuseNeurals)



