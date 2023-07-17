import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def detect_anomalies(data):
    # Initialize the LocalOutlierFactor model with novelty=True to detect anomalies
    lof_model = LocalOutlierFactor(novelty=True)

    # Fit the model to the data (data without anomalies)
    lof_model.fit(data)

    # Detect outliers (anomalies) in the data
    outliers = lof_model.predict(data)

    # Get the indices of the detected anomalies
    anomaly_indices = np.where(outliers == -1)[0]

    # Get the data points corresponding to the detected anomalies
    anomalies = data[anomaly_indices]

    return anomalies
