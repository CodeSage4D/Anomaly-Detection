# # Import necessary functions from data_loader and anomaly_detector modules
# from data_loader import load_data
# from anomaly_detector import detect_anomalies
# import os
#
# print("Current working directory:", os.getcwd())
#
# # Step 1: Load Data
# # Call the load_data function from data_loader module to load data from 'data.csv'
# data = load_data('D:\Internshala ML\Datasets\datatest.csv')
#
# # Step 2: Handle Missing Values (Optional)
# # Uncomment one of the following approaches:
# # Approach 1: Remove Rows with Missing Values
# # import numpy as np
# # data = data[~np.isnan(data).any(axis=1)]
#
# # Approach 2: Impute Missing Values
# # from sklearn.impute import SimpleImputer
# # imputer = SimpleImputer(strategy='mean')
# # data = imputer.fit_transform(data)
#
# # Step 3: Anomaly Detection
# # Call the detect_anomalies function from anomaly_detector module to detect anomalies in the loaded data
# anomalies = detect_anomalies(data)
#
# # Step 4: Save Anomalies to a File
# # Open 'anomalies.txt' file in write mode ('w')
# with open('anomalies.txt', 'w') as file:
#     # Write a header indicating that the file contains detected anomalies
#     file.write("# Detected Anomalies:\n")
#     # Iterate through each detected anomaly
#     for anomaly in anomalies:
#         # Join the values in the anomaly tuple into a comma-separated string and write it to the file
#         file.write(','.join(str(val) for val in anomaly) + '\n')
#
# # Print a message indicating that anomalies have been detected and saved to the file
# print("Anomalies detected and saved to 'anomalies.txt'")

import numpy as np
from data_loader import load_data
from anomaly_detector import detect_anomalies
import os

print("Current working directory:", os.getcwd())

# Step 1: Load Data
data = load_data('D:\Internshala ML\Datasets\datatest.csv')

# Step 2: Anomaly Detection
anomalies = detect_anomalies(data)

# Step 3: Save Anomalies to a File
with open('anomalies.txt', 'w') as file:
    file.write("# Detected Anomalies:\n")
    for anomaly in anomalies:
        file.write(','.join(str(val) for val in anomaly) + '\n')

print("Anomalies detected and saved to 'anomalies.txt'")
