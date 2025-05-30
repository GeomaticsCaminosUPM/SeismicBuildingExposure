from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

from tabpfn import TabPFNClassifier

df = pd.read_csv("3_classes_processed_data.csv")
label_encoders = {}
for column in ['ductility', 'roof', 'relative_position']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Encode the target variable
label_encoder_target = LabelEncoder()
df['structural_system'] = label_encoder_target.fit_transform(df['structural_system'])

# Define features and target
X = df.drop(columns=['structural_system'])
y = df['structural_system']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Convert to PyTorch tensors
# X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
# X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Initialize a classifier
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)

# Predict labels
predictions = clf.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, predictions))

# Calculate ROC AUC for multi-class classification using One-vs-Rest (ovr)
roc_auc = roc_auc_score(y_test, prediction_probabilities, multi_class='ovr')
print("\nROC AUC:", roc_auc)

# Print Accuracy
print("\nAccuracy:", accuracy_score(y_test, predictions))
