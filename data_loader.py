import numpy as np
from sklearn.impute import SimpleImputer


def load_data(file_name):
    # Load the dataset from 'file_name' using NumPy or any other data loading technique.
    # For simplicity, we will use a NumPy array here.
    # np.genfromtxt is used to read data from 'file_name' and split it by commas (delimiter=',')
    data = np.genfromtxt(file_name, delimiter=',')

    # Check if any columns have all missing values (NaN)
    missing_columns = np.isnan(data).all(axis=0)
    columns_to_keep = ~missing_columns

    # Remove the columns with all missing values
    data = data[:, columns_to_keep]

    # Replace NaN values in the remaining columns with the mean of the respective columns
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)

    # Return the loaded and preprocessed data
    return data
