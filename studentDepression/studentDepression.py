import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load and preprocess the data
df = pd.read_csv('studentDepression.csv')

# Data exploration
print("Dataset info:")
print(f"Total records: {df.shape[0]}")
print(f"Depression distribution: \n{df['Depression'].value_counts()}")

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values:")
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")

# Drop unnecessary columns
df = df.drop(['id', 'City'], axis=1)

# Encode categorical variables
categorical_columns = df.select_dtypes(include='object').columns
le = LabelEncoder()
for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

# Features and target
X = df.drop(['Depression'], axis=1)
y = df['Depression']

# Split the data before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling - fit only on training data to prevent data leakage
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Check class distribution
print("\nClass distribution before SMOTE:")
print(f"Training set: \n{y_train.value_counts()}")

# SMOTE for balancing - apply only to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Check class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(f"Training set: \n{pd.Series(y_train_resampled).value_counts()}")

# Train the model - using Random Forest Classifier as requested
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    criterion='gini',
    random_state=42
)
rf.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = rf.predict(X_test_scaled)

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
plt.savefig('confusion_matrix.png')
plt.close()
print("Confusion matrix saved as 'confusion_matrix.png'")

# Feature importance
feature_importance = pd.DataFrame(
    {'Feature': X_train.columns, 'Importance': rf.feature_importances_}
)
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("Feature importance plot saved as 'feature_importance.png'")

# Save the model
with open('student_depression_model.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

# Save the scaler
with open('student_depression_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")