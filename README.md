This project is dedicated to an example of using KNN for regression and classification problems. The small data set is a subset of
Iris data. You can find the complete data set here: https://www.kaggle.com/uciml/iris?select=Iris.csv. If you have multiple csv files, keep all of them in the Data directory. 


## What is in this project?
- dataset.py contains the aggregation of all the csv files in ""Data"" directory and encoding the labels of the aggregated data.
- encoder.obj is an encoder, which is fitted on the data and saved in order to be used for future data.
- model.py includes a KNN classifier which is built after splitting the aggregated data in to test and train.
- selectK.py is a function for choosing the k with respect to the error values. It visualizes the error rates versus k values.
