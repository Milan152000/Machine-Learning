import pandas as pd
import numpy as np
import joblib
from sklearn import svm,metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Loading the dataset and make dataframe and print dataframe...............
dataframe = pd.read_csv("D:\\My Project\\csv\\dataset.csv")
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
print(dataframe)

# Splitting of Training and Testing Data................
X = dataframe.drop(['label'], axis=1)
Y = dataframe['label']

X_train , Y_train = X[0:300] , Y[0:300]
X_test , Y_test = X[300:] , Y[300:]

print(X_train.values[40].reshape(28,28))

# Plot the sample of one datapoint with its label..................
# grid_data = X_train.values[40].reshape(28,28)
# plt.imshow(grid_data,interpolation=None,cmap="gray")
# plt.title(Y_train.values[40])
# plt.show()

# Creating the SVM model and storing it.................
model = svm.SVC(kernel="linear",C=10)
model.fit(X_train,Y_train)
joblib.dump(model, "D:\\My Project\\model\\svm_0to9_model_linear") 

# Predictions by the model and getting accuracy score..............
model = joblib.load("D:\\My Project\\model\\svm_0to9_model_linear")
print ("predicting .....")
predictions = model.predict(X_test)
print(predictions)

print ("Getting Accuracy .....")
print ("Score", metrics.accuracy_score(Y_test, predictions)*100)
