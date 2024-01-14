import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import functorch
from torch.func import grad, vmap, functional_call
import argparse
import torch.nn.functional as F

from grabsampler import GraBSampler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
features_tuple = torch.load("data/features-processed-NY-2017.pt")
targets_tuple = torch.load("data/targets-processed-NY-2017.pt")

features = features_tuple[0].to("cpu")
targets = targets_tuple[0].to("cpu")

features_np = features.numpy()
targets_np = targets.numpy()

scaler = StandardScaler()
features_np = scaler.fit_transform(features_np)

features = torch.tensor(features_np, dtype=torch.float32)
targets = torch.tensor(targets_np, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=42
)


class LogisticRegression(nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)


model = LogisticRegression(features.shape[1])
model.to(device)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Logistic Regression with functorch")
parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument(
    "--accumulation_steps", type=int, default=1, help="Gradient accumulation steps"
)

args = parser.parse_args()

# Use the arguments
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
accumulation_steps = args.accumulation_steps

# Convert model to functional form
# fmodel, params, buffers = functorch.make_functional_with_buffers(model)
params = dict(model.named_parameters())
buffers = dict(model.named_buffers())

for p in params.values():
    p.requires_grad = False

# DataLoader
train_data = torch.utils.data.TensorDataset(X_train, y_train)

sampler = GraBSampler(train_data, params, balance_type="mean")

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, sampler=sampler
)


# Define the loss function for the functional model
def compute_loss_stateless_model(params, buffers, X, y):
    yhat = functional_call(model, (params, buffers), (X,)).squeeze()
    return F.binary_cross_entropy_with_logits(yhat, y.squeeze())


# Define a function to compute per-example gradients
func_per_example_grad = vmap(
    grad(compute_loss_stateless_model), in_dims=(None, None, 0, 0)
)

# Training Loop
with torch.no_grad():
    for e in range(1, args.epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Compute per-example gradients
            per_example_grads = func_per_example_grad(params, buffers, X_batch, y_batch)

            sampler.step(per_example_grads)

            # Average the gradients over the batch
            avg_grads = [torch.mean(g, dim=0) for g in per_example_grads.values()]

            # Update parameters using averaged gradients
            with torch.no_grad():
                for p, g in zip(params.values(), avg_grads):
                    p -= learning_rate * g

        print(f"Epoch {e}/{epochs} complete")

        # Evaluation
        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            # logits = fmodel(params, buffers, X_test).squeeze()
            logits = model(X_test).squeeze()
            # y_pred = torch.sigmoid(logits)
            # y_pred = (y_pred >= 0.5).float()
            y_pred = torch.sign(logits)
            accuracy = (y_pred == y_test).float().mean()
            print(f"Accuracy: {accuracy:.4f}")

        del X_test, y_test
