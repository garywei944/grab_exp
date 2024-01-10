import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

df = pd.read_csv('/home/dyc33/grab_exp/grab_exp/data/hmda_2017_ny_first-lien-owner-occupied-1-4-family-records_labels.csv')

mixed_type_cols = ['applicant_ethnicity_name', 'applicant_race_name_1', 'co_applicant_ethnicity_name', 'co_applicant_race_name_1']
df[mixed_type_cols] = df[mixed_type_cols].astype(str)
df.fillna('Missing', inplace=True) 

target = 'action_taken'
X = df.drop(target, axis=1)
y = df[target]

# Specify actual categorical columns
categorical_features = [
    'agency_name', 'loan_type_name', 'property_type_name', 
    'loan_purpose_name', 'applicant_ethnicity_name', 'applicant_race_name_1', 
    'applicant_sex_name', # Add more categorical columns as needed
]

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features)])

X = preprocessor.fit_transform(X)
y = y.values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Discard N mod B examples from X_train and y_train if necessary
batch_size = 32  # Adjust as needed
remainder = X_train.shape[0] % batch_size
if remainder != 0:
    X_train = X_train[:-remainder]
    y_train = y_train[:-remainder]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Logistic Regression with PyTorch
class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

n_features = X_train.shape[1]
model = LogisticRegressionPyTorch(n_features)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()

# Predictions
# Evaluation
with torch.no_grad():
    y_predicted = model(X_test).squeeze()
    y_predicted_cls = y_predicted.round()
    acc = accuracy_score(y_test, y_predicted_cls)
    print(f"Accuracy: {acc}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_predicted_cls))
    print("Classification Report:\n", classification_report(y_test, y_predicted_cls))
    print(df['action_taken'].value_counts())


