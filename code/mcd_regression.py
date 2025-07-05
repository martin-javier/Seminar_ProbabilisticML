# Uncertainty estimation using MC-Dropout on an MLP model for regression
# Model is trained on a synthetic regression task, and MC-Dropout inference is performed to obtain predictive mean and uncertainty estimates.

# Packages needed: torch, numpy, sklearn (scikit-learn), matplotlib, skorch (later for tuning)


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(7)
torch.manual_seed(7)

##########################################################################################
#
# 1. Dataset
#
##########################################################################################

N = 1000
x = np.linspace(-5, 5, N)
y = np.sin(x) + 0.3 * np.random.randn(N)  # underlying sin curve + Gaussian noise

# Split into train/test
x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=7)

# Create test set for extrapolation
x_test = np.linspace(-8, 8, 100)
y_test = np.sin(x_test)  # true function for reference

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)


##########################################################################################
#
# 2. Model
#
##########################################################################################

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2, n_out, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden2, n_out),
        )
    def forward(self, x):
        return self.net(x)

# Set hyperparameters
epochs = 1000
lr = 1e-3
wd = 1e-5
batch = 20
mc_samples = 200

# Build DataLoader
train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch, shuffle=True)


##########################################################################################
#
# 3. MC-Dropout helper function
#
##########################################################################################

def mc_dropout_predict(model, x, n_samples=100):
    """
    Run `n_samples` stochastic forward passes with dropout active,
    and return (mean, variance) across those passes.
    """
    model.train()  # ensures dropout is active during inference
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x).cpu().numpy())
    preds = np.stack(preds, axis=0)
    mean = preds.mean(axis=0)
    var = preds.var(axis=0)
    return mean, var


##########################################################################################
#
# 4. Training and testing
#
##########################################################################################

# Initialize model, optimizer, and loss
model = MLP(1, 32, 32, 1, dropout_rate=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = torch.nn.MSELoss()

# Training loop
model.train()
for epoch in range(epochs):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"[Epoch {epoch+1}/{epochs}]  loss: {loss.item():.4f}")

# MC-Dropout inference on test set
preds = []
model.train()  # keep dropout active
with torch.no_grad():
    for _ in range(mc_samples):
        preds.append(model(x_test_tensor).cpu().numpy())
preds = np.stack(preds, axis=0)
mean_preds = preds.mean(axis=0).flatten()
std_preds = preds.std(axis=0).flatten()

##########################################################################################
#
# 5. Plotting
#
##########################################################################################

plt.style.use('default')
plt.figure(figsize=(19.2, 10.8))

# True function
plt.plot(x_test, np.sin(x_test), linestyle='--', color='black', linewidth=3, label='True function')

# MC samples
for i in range(preds.shape[0]):
    plt.scatter(x_test, preds[i,:,0], s=10, alpha=0.5, color='lightblue', label='MC samples' if i == 0 else "")

# Mean prediction
plt.plot(x_test, mean_preds, color='crimson', linewidth=3, label='Mean prediction')

# Uncertainty band
plt.fill_between(x_test, mean_preds - 2*std_preds, mean_preds + 2*std_preds, color='crimson', alpha=0.3, label='Uncertainty (±2 std)')

# Training region
plt.axvline(-5, linestyle=':', color='grey', linewidth=2)
plt.axvline(5, linestyle=':', color='grey', linewidth=2)
plt.text(0, -2.5, 'Training region', color='grey', fontsize=22, ha='center')

# Labels & legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=3, label='True function'),
    Line2D([0], [0], marker='o', color='lightblue', markersize=10, linestyle='None', label='MC samples', alpha=1.0),
    Line2D([0], [0], color='crimson', linewidth=3, label='Mean prediction'),
    Line2D([0], [0], color='crimson', linewidth=10, alpha=0.3, label='Uncertainty (±2 std)')
]

plt.title('MC-Dropout: Uncertainty Estimation on a Regression Task', fontsize=34, pad=15)
plt.xlabel('x', fontsize=26)
plt.ylabel('y', fontsize=26)
plt.legend(handles=legend_elements, loc='upper right', fontsize=28)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.grid(alpha=0.3)
plt.xlim(-8.3, 8.3)
plt.ylim(-3.25, 3.25)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Auto-save the plot
plt.savefig("plots/mcd_regression.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# Why are we seeing vertical "pillars" of samples? It's exactly what we want:
# - The test set (x_test) consists of 100 equidistant points in the interval [-8, 8] (every pillar is on points x coordinate)
# - For every test point, 200 MC samples are drawn, so we get 200 blue dots stacked vertically at that x
# - The mean prediction (red line) is the average of these samples, and the shaded area represents the uncertainty (±2 standard deviations)
# - The unceartainty increases in the extrapolation region (outside the training data) -> OOD Unceartainty "Out-of-Distribution Uncertainty"




# The same process but with tuning of the MLP hyperparameters using skorch and RandomizedSearchCV:

""" import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D """
from skorch import NeuralNetRegressor
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

np.random.seed(7)
torch.manual_seed(7)

##########################################################################################
#
# 1. Dataset
#
##########################################################################################

N = 1000
x = np.linspace(-5, 5, N)
y = np.sin(x) + 0.3 * np.random.randn(N)  # underlying sin curve + Gaussian noise

# Split into train+validation (80%) and test (20%)
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
# Further split train+validation into train (75% of 80% = 60% of total) and validation (25% of 80% = 20% of total)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=7)

# Create test set
x_test_extrap = np.linspace(-8, 8, 100)
y_test_extrap = np.sin(x_test_extrap)  # true function for reference and plotting

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
x_test_extrap_tensor = torch.tensor(x_test_extrap, dtype=torch.float32).unsqueeze(1)


##########################################################################################
#
# 2. Model
#
##########################################################################################

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2, n_out, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden2, n_out),
        )
    def forward(self, x):
        return self.net(x)


##########################################################################################
#
# 3. Hyperparameter Tuning
#
##########################################################################################

# Prepare data for tuning
X_tuning = x_train.reshape(-1, 1).astype(np.float32)
y_tuning = y_train.reshape(-1, 1).astype(np.float32)
X_val = x_val.reshape(-1, 1).astype(np.float32)
y_val_arr = y_val.reshape(-1, 1).astype(np.float32)

# Wrap model with skorch
net = NeuralNetRegressor(
    module=MLP,
    module__n_in=1,
    module__n_out=1,
    max_epochs=300,
    optimizer=torch.optim.Adam,
    criterion=torch.nn.MSELoss,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    train_split=None,
    verbose=0
)

# Define parameter distribution for randomized search
param_dist = {
    'lr': [1e-4, 1e-3, 1e-2],
    'optimizer__weight_decay': [0, 1e-5, 1e-4, 1e-3],
    'batch_size': [16, 32, 64],
    'module__n_hidden1': [32, 64, 128],
    'module__n_hidden2': [32, 64, 128],
    'module__dropout_rate': [0.1, 0.2, 0.3],
}

# Create split: -1 for train, 0 for validation
split_index = [-1] * len(X_tuning) + [0] * len(X_val)
ps = PredefinedSplit([split_index])

# Combine train and validation data
X_combined = np.vstack((X_tuning, X_val))
y_combined = np.vstack((y_tuning, y_val_arr))

# Randomized search with validation set
random_search = RandomizedSearchCV(
    net, 
    param_dist, 
    n_iter=10,
    cv=ps,
    scoring='neg_mean_squared_error',
    random_state=7,
    n_jobs=-1,
    verbose=1
)

# Run hyperparameter search
print("Starting hyperparameter tuning with train/validation split...")
random_search.fit(X_combined, y_combined)
print("Tuning complete!")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best validation MSE: {-random_search.best_score_:.4f}")

# Extract best parameters
best_params = random_search.best_params_


##########################################################################################
#
# 4. Final Training (on combined train + validation set)
#
##########################################################################################

# Set hyperparameters from tuning results
epochs = 1000
lr = best_params['lr']
wd = best_params['optimizer__weight_decay']
batch = best_params['batch_size']
n_hidden1 = best_params['module__n_hidden1']
n_hidden2 = best_params['module__n_hidden2']
dropout_rate = best_params['module__dropout_rate']
wd = 1e-5
mc_samples = 200

# Combine train and validation sets for final training
x_train_full = np.concatenate([x_train, x_val])
y_train_full = np.concatenate([y_train, y_val])
x_train_full_tensor = torch.tensor(x_train_full, dtype=torch.float32).unsqueeze(1)
y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)

# Build DataLoader for combined dataset
train_loader_full = DataLoader(
    TensorDataset(x_train_full_tensor, y_train_full_tensor), 
    batch_size=batch, 
    shuffle=True
)

# Initialize model with best hyperparameters
model = MLP(1, n_hidden1, n_hidden2, 1, dropout_rate=dropout_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = torch.nn.MSELoss()

# Training loop on full train+val set
model.train()
for epoch in range(epochs):
    for xb, yb in train_loader_full:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"[Epoch {epoch+1}/{epochs}]  loss: {loss.item():.4f}")


##########################################################################################
#
# 5. Final Testing (on prev unseen test set)
#
##########################################################################################

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_pred = model(x_test_tensor)
    test_mse = criterion(test_pred, torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
print(f"\nFinal test MSE (in-distribution): {test_mse.item():.4f}")

# MC-Dropout inference on extrapolation set
preds = []
model.train()  # keep dropout active
with torch.no_grad():
    for _ in range(mc_samples):
        preds.append(model(x_test_extrap_tensor).cpu().numpy())
preds = np.stack(preds, axis=0)
mean_preds = preds.mean(axis=0).flatten()
std_preds = preds.std(axis=0).flatten()


##########################################################################################
#
# 6. Plotting
#
##########################################################################################

plt.style.use('default')
plt.figure(figsize=(19.2, 10.8))

# True function
plt.plot(x_test_extrap, np.sin(x_test_extrap), linestyle='--', color='black', linewidth=3, label='True function')

# MC samples
for i in range(min(100, preds.shape[0])):  # Plot max 100 samples for clarity
    plt.scatter(x_test_extrap, preds[i,:,0], s=10, alpha=0.5, color='lightblue', label='MC samples' if i == 0 else "")

# Mean prediction
plt.plot(x_test_extrap, mean_preds, color='crimson', linewidth=3, label='Mean prediction')

# Uncertainty band
plt.fill_between(x_test_extrap, mean_preds - 2*std_preds, mean_preds + 2*std_preds, color='crimson', alpha=0.3, label='Uncertainty (±2 std)')

# Training region
plt.axvline(-5, linestyle=':', color='grey', linewidth=2)
plt.axvline(5, linestyle=':', color='grey', linewidth=2)
plt.text(0, -2.5, 'Training region', color='grey', fontsize=22, ha='center')

# Labels & legend
legend_elements = [
    Line2D([0], [0], color='black', linestyle='--', linewidth=3, label='True function'),
    Line2D([0], [0], marker='o', color='lightblue', markersize=10, linestyle='None', label='MC samples', alpha=1.0),
    Line2D([0], [0], color='crimson', linewidth=3, label='Mean prediction'),
    Line2D([0], [0], color='crimson', linewidth=10, alpha=0.3, label='Uncertainty (±2 std)')
]
plt.title('MC-Dropout: Uncertainty Estimation on a Regression Task (tuned NN)', fontsize=34, pad=15)
plt.xlabel('x', fontsize=26)
plt.ylabel('y', fontsize=26)
plt.legend(handles=legend_elements, loc='upper right', fontsize=28)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(alpha=0.3)
plt.xlim(-8.3, 8.3)
plt.ylim(-3.25, 3.25)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Auto-save the plot
plt.savefig("plots/mcd_reg_tuned.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()