import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset
data = pd.read_csv("Datasets/blood_usage_data.csv")

# Handle missing values
data.fillna(0, inplace=True)

# Encode categorical variables (e.g., blood type)
encoder = OneHotEncoder()
encoded_blood_type = encoder.fit_transform(data[['blood_type']]).toarray()
encoded_blood_type = pd.DataFrame(encoded_blood_type, columns=encoder.get_feature_names_out(['blood_type']))

# Normalize numerical features
scaler = StandardScaler()
data[['demand', 'inventory']] = scaler.fit_transform(data[['demand', 'inventory']])

# Combine encoded and normalized data
processed_data = pd.concat([data[['date', 'demand', 'inventory']], encoded_blood_type], axis=1)

# Save processed data
processed_data.to_csv("Datasets/processed_blood_data.csv", index=False)