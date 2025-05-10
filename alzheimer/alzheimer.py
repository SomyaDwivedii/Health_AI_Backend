import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
# from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

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

y_pred = xgboost_model.predict(X_test)
# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model
with open('alzheimer_model.pkl', 'wb') as model_file:
    pickle.dump(xgboost_model, model_file)

# Save the scaler
with open('alzheimer_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Alzheimer's model and scaler saved successfully!")