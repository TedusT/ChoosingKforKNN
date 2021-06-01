from sklearn import neighbors
import numpy as np
import pandas as pd
# from sklearn.metrics import mean_squared_error
# from math import sqrt

def select_k(predictors_train, y_train, predictors_test, y_test):
    # Calculate for all k, the error values
    error_values = []
    for K in range(20):
        K = K + 1
        # For regression use the following model

        # model = neighbors.KNeighborsRegressor(n_neighbors = K)
        # For classification use this
        model = neighbors.KNeighborsClassifier(n_neighbors=K)
        model.fit(predictors_train, y_train)
        # predict the duration on the test set
        prediction = model.predict(predictors_test)

        # For regression get RMSE values
        # error_rate = sqrt(mean_squared_error(y_te,prediction))

        # For classification
        error_rate = np.mean(prediction != y_test)

        # Add each error rate to the first error values list
        error_values.append(error_rate)
    # Make a data frame including errors and k values
    k_error_values = pd.DataFrame(error_values)
    k_error_values = k_error_values.assign(K=range(20))
    k_error_values.columns = ["error", "K"]

    return k_error_values