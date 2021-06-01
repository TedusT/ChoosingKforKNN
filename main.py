# Import necessary packages

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from selectK import select_k
import pandas as pd
from dataset import preprocessdata
import matplotlib.pyplot as plt
import numpy as np


# Get the preprocessed, encoded data
dataset = preprocessdata()
encoded_data = dataset.get_set()

# The data is splitted into train and test data, the size of test is 25% of the actual data.

train_data , test_data = train_test_split(encoded_data, test_size = 0.25)

# All the features except Class are used as predictors
x_train_final = train_data.drop('Class', axis=1)
y_train_final = train_data['Class']

x_test_final = test_data.drop('Class', axis = 1)
y_test_final = test_data['Class']

# All the features are scaled between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train_final)
x_train_scaled = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test_final)
x_test_scaled = pd.DataFrame(x_test_scaled)

# For testing on any unseen data, the encoder should be called here before any prediction.
# file = open("encoder.obj", 'rb')
# encoder_loaded = pickle.load(file)
# file.close()
# new_data_encoded=encoder_loaded.transform(new_data)


if __name__ == "__main__":
    # Call the select_k function with all the data splits as inputs
    k_error_values = select_k(x_train_scaled, y_train_final, x_test_scaled, y_test_final)
    selected_k = k_error_values.loc[k_error_values['error'] == min(k_error_values.error), 'K'].iloc[0]
    min_error = k_error_values.loc[k_error_values['error'] == min(k_error_values.error), 'error'].iloc[0]

    # Visualize the error values and k to find out at which k, the error rates are low
    plt.plot(k_error_values['K'], k_error_values['error'], color='green', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
    plt.xlabel('K', fontsize=18)
    plt.ylabel('Mean Error', fontsize=16)
    plt.show()
    print('Chosen k=', selected_k, 'min error=', min_error)
