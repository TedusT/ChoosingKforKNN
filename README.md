This project is dedicated to an example of using KNN for regression and classification problems. The data set is 
Iris Species data. You can find the data set here: https://www.kaggle.com/uciml/iris?select=Iris.csv


## What is in this project?
- dataset.py contains the aggregation of all the csv files in ""Data"" directory and encoding the labels of the aggregated data.
- encoder.obj is an encoder, which is fitted on the data and saved in order to be used for future data.
- model.py includes a KNN classifier which is built after splitting the aggregated data in to test and train.
- selectK.py is a function for choosing the k with respect to the error values. It visualizes the error rates versus k values.