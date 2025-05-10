import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE

# Load and preprocess the data (similar to your original script)
df = pd.read_csv('diabetes.csv')

# Data preprocessing steps
columns_non_nol = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df1 = df.copy()

df1[columns_non_nol] = df1[columns_non_nol].replace(0, np.nan)

for column in columns_non_nol:
    median_value = df1[column].median()
    df1[column].fillna(median_value, inplace=True)

# Features and target
X = df1.drop(['Outcome'], axis=1)
y = df1['Outcome']

# Scaling
features_to_scale = X.columns
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# SMOTE for balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train the model
et = ExtraTreesClassifier(
    n_estimators=200, 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    bootstrap=False, 
    criterion='entropy'
)
et.fit(X_resampled, y_resampled)

# Save the model
with open('diabetes_model.pkl', 'wb') as model_file:
    pickle.dump(et, model_file)

# Save the scaler
with open('diabetes_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")