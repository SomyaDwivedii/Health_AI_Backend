import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
# from catboost import CatBoostClassifier
from xgboost import XGBClassifier
# Load and preprocess the data
df = pd.read_csv('alzheimers_disease_data.csv')

# Drop unnecessary columns
df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

# Define columns to normalize
columns_to_normalize = [
    'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 
    'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP', 
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 
    'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL'
]

# Split data into features and target
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Apply Min-Max scaling to the columns
scaler = MinMaxScaler()
X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Train the CatBoost model with optimal parameters
# catboost_model = CatBoostClassifier(
#     iterations=100,
#     learning_rate=0.1,
#     verbose=0
# )
# catboost_model.fit(X_train, y_train)

xgboost_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    verbose=0
)
xgboost_model.fit(X_train, y_train)

# Save the model
with open('alzheimer_model.pkl', 'wb') as model_file:
    pickle.dump(xgboost_model, model_file)

# Save the scaler
with open('alzheimer_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Alzheimer's model and scaler saved successfully!")