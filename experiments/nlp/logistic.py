import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
features_tuple = torch.load('/home/dyc33/grab_exp/grab_exp/data/features-processed-NY-2017.pt')
targets_tuple  = torch.load('/home/dyc33/grab_exp/grab_exp/data/targets-processed-NY-2017.pt')

features = features_tuple[0].to('cpu')
targets = targets_tuple[0].to('cpu')

features_np = features.numpy()
targets_np = targets.numpy()

scaler = StandardScaler()
features_np = scaler.fit_transform(features_np)

features = torch.tensor(features_np, dtype=torch.float32)
targets = torch.tensor(targets_np, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

class LogisticRegression(nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)  

model = LogisticRegression(features.shape[1])
model.to(device)

learning_rate = 1e-2
batch_size = 16
epochs = 50
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss()  

train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# Training Loop
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits = model(X_batch).squeeze()
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    logits = model(X_test).squeeze()
    y_pred = torch.sigmoid(logits)  
    y_pred = (y_pred >= 0.5).float()
    accuracy = (y_pred == y_test).float().mean()
    print(f'Accuracy: {accuracy:.4f}')
