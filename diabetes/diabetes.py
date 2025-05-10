import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load and preprocess the data
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

# Split the data before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling - fit only on training data to prevent data leakage
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# SMOTE for balancing - apply only to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train the model
et = ExtraTreesClassifier(
    n_estimators=200, 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    bootstrap=False, 
    criterion='entropy'
)
et.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = et.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Generate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Save the model
with open('diabetes_model.pkl', 'wb') as model_file:
    pickle.dump(et, model_file)

# Save the scaler
with open('diabetes_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")