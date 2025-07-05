# Uncertainty estimation using MC-Dropout on an MLP model for classification
# Synthetic dataset generated using make_moons from sklearn.
# Model is trained on classification task, MC-Dropout inference is performed to obtain predictive mean and uncertainty estimates.

# Packages needed: torch, numpy, sklearn (scikit-learn), matplotlib, skorch (later for tuning)


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from matplotlib.colors import TwoSlopeNorm

np.random.seed(7)
torch.manual_seed(7)

##########################################################################################
#
# 1. Dataset
#
##########################################################################################

X, y = make_moons(n_samples=500, noise=0.2)

# Evaluation grid for visualization
xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
X_test_grid = np.c_[xx.ravel(), yy.ravel()]

X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_grid, dtype=torch.float32)


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
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Set hyperparameters
epochs = 300  # epochs
lr = 1e-2  # learning rate 
wd = 1e-4  # weight decay
batch = 32
mc_samples = 200  # number of MC-Dropout samples

# Create DataLoader for training
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch, shuffle=True)


##########################################################################################
#
# 3. MC-Dropout helper function
#
##########################################################################################

# Helper for MC Dropout predictions
def mc_dropout_predict(model, x, n_samples=100):
    model.train()  # keep dropout active
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

# Initialize model, optimizer, loss
model = MLP(2, 32, 32, 1, dropout_rate=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = nn.BCELoss()

# Train the model
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Run MC Dropout
mean_preds, var_preds = mc_dropout_predict(model, X_test_tensor, n_samples=mc_samples)


##########################################################################################
#
# 5. Plotting
#
##########################################################################################

plt.style.use('default')
fig, axs = plt.subplots(1, 2, figsize=(19.2, 10.8))

# Custom diverging colormap centered at 0.5 for binary classification
cmap = plt.get_cmap("bwr")
norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

# A) Predictive mean
cf1 = axs[0].contourf(xx, yy, mean_preds.reshape(xx.shape), levels=100, cmap=cmap, norm=norm, alpha=0.8)
axs[0].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=50)
axs[0].set_title("MC-Dropout: Prediction Probability", fontsize=34, pad=15)
axs[0].set_xlabel("x1", fontsize=26)
axs[0].set_ylabel("x2", fontsize=26)
axs[0].tick_params(labelsize=22)
cbar1 = fig.colorbar(cf1, ax=axs[0])
cbar1.ax.tick_params(labelsize=22)

# B) Uncertainty heatmap
uncertainty = np.sqrt(var_preds).reshape(xx.shape)
cf2 = axs[1].contourf(xx, yy, uncertainty, levels=100, cmap="viridis")
axs[1].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=50)
axs[1].set_title("Predictive Uncertainty", fontsize=34, pad=15)
axs[1].set_xlabel("x1", fontsize=26)
axs[1].set_ylabel("x2", fontsize=26)
axs[1].tick_params(labelsize=22)
cbar2 = fig.colorbar(cf2, ax=axs[1])
cbar2.ax.tick_params(labelsize=22)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Auto-save the plot
plt.savefig("plots/mcd_classification.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# The plot shows the decision boundary and uncertainty of the model.
# The left subplot displays the predicted probabilities for the positive class, while the right subplot shows the predictive uncertainty (standard deviation) across the input space.
# The model is trained on a synthetic binary classification task using the "moons" dataset, which is a common benchmark for testing classification algorithms.
# We can visually interpret uncertainty from regions where the backgroud is white or pale -> model is unsure between classes.
# Regions with strong color indicate high confidence in the prediction.



# The same process but with tuning of the MLP hyperparameters using skorch and RandomizedSearchCV:

""" import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from matplotlib.colors import TwoSlopeNorm """
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

np.random.seed(7)
torch.manual_seed(7)

##########################################################################################
#
# 1. Dataset
#
##########################################################################################

# Generate moon dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=7)

# Split into train+val (80%) and test (20%)
X_train_val, X_test_holdout, y_train_val, y_test_holdout = train_test_split(
    X, y, test_size=0.2, random_state=7
)

# Further split train+val into train (64%) and validation (16%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=7
)

# Evaluation grid for visualization
xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
X_test_grid = np.c_[xx.ravel(), yy.ravel()]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_grid, dtype=torch.float32)
X_test_holdout_tensor = torch.tensor(X_test_holdout, dtype=torch.float32)


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
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


##########################################################################################
#
# 3. Hyperparameter Tuning
#
##########################################################################################

# Prepare data for tuning
X_tuning = X_train.astype(np.float32)
y_tuning = y_train.astype(np.float32).reshape(-1, 1)
X_val_arr = X_val.astype(np.float32)
y_val_arr = y_val.astype(np.float32).reshape(-1, 1)

# Wrap model with skorch
net = NeuralNetClassifier(
    module=MLP,
    module__n_in=2,
    module__n_out=1,
    max_epochs=300,
    optimizer=torch.optim.Adam,
    criterion=nn.BCELoss,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    train_split=None,
    verbose=0
)

# Define parameter distribution for randomized search
param_dist = {
    'lr': [1e-3, 1e-2, 5e-2],
    'optimizer__weight_decay': [0, 1e-5, 1e-4, 1e-3],
    'batch_size': [16, 32, 64],
    'module__n_hidden1': [16, 32, 64],
    'module__n_hidden2': [16, 32, 64],
    'module__dropout_rate': [0.1, 0.2, 0.3, 0.4],
}

# Predefined split for validation set
test_fold = np.array([-1] * len(X_tuning) + [0] * len(X_val_arr))
ps = PredefinedSplit(test_fold)

# Combine train and validation data
X_combined = np.vstack((X_tuning, X_val_arr))
y_combined = np.vstack((y_tuning, y_val_arr))

# Randomized search with validation set
random_search = RandomizedSearchCV(
    net, 
    param_dist, 
    n_iter=10,
    cv=ps,
    scoring='accuracy',
    random_state=7,
    n_jobs=-1,
    verbose=1
)

# Run hyperparameter search
print("Starting hyperparameter tuning with train/validation split...")
random_search.fit(X_combined, y_combined)
print("Tuning complete!")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best validation accuracy: {random_search.best_score_:.4f}")

# Extract best parameters
best_params = random_search.best_params_


##########################################################################################
#
# 4. Training and Testing
#
##########################################################################################

# Set hyperparameters from tuning results
epochs = 300
lr = best_params['lr']
wd = best_params['optimizer__weight_decay']
batch = best_params['batch_size']
n_hidden1 = best_params['module__n_hidden1']
n_hidden2 = best_params['module__n_hidden2']
dropout_rate = best_params['module__dropout_rate']
wd = 1e-4
mc_samples = 200

# Combine train and validation sets for final training
X_train_full = np.concatenate([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])
X_train_full_tensor = torch.tensor(X_train_full, dtype=torch.float32)
y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)

# Create DataLoader for training
train_loader = DataLoader(
    TensorDataset(X_train_full_tensor, y_train_full_tensor), 
    batch_size=batch, 
    shuffle=True
)

# Initialize model with best hyperparameters
model = MLP(2, n_hidden1, n_hidden2, 1, dropout_rate=dropout_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = nn.BCELoss()

# Train the model
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluate on held-out test set
model.eval()
with torch.no_grad():
    test_preds = model(X_test_holdout_tensor)
    test_preds = (test_preds > 0.5).float()
    test_acc = (test_preds.squeeze() == torch.tensor(y_test_holdout, dtype=torch.float32)).float().mean()
print(f"\nFinal test accuracy: {test_acc.item():.4f}")

# MC-Dropout helper function
def mc_dropout_predict(model, x, n_samples=100):
    model.train()  # keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x).cpu().numpy())
    preds = np.stack(preds, axis=0)
    mean = preds.mean(axis=0)
    var = preds.var(axis=0)
    return mean, var

# Run MC Dropout on visualization grid
mean_preds, var_preds = mc_dropout_predict(model, X_test_tensor, n_samples=mc_samples)


##########################################################################################
#
# 5. Plotting
#
##########################################################################################

plt.style.use('default')
fig, axs = plt.subplots(1, 2, figsize=(19.2, 10.8))

# Custom diverging colormap centered at 0.5 for binary classification
cmap = plt.get_cmap("bwr")
norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

# A) Predictive mean
cf1 = axs[0].contourf(xx, yy, mean_preds.reshape(xx.shape), levels=100, cmap=cmap, norm=norm, alpha=0.8)
axs[0].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=50)
axs[0].set_title("MC-Dropout: Prediction Probability", fontsize=34, pad=15)
axs[0].set_xlabel("x1", fontsize=26)
axs[0].set_ylabel("x2", fontsize=26)
axs[0].tick_params(labelsize=22)
cbar1 = fig.colorbar(cf1, ax=axs[0])
cbar1.ax.tick_params(labelsize=22)

# B) Uncertainty heatmap
uncertainty = np.sqrt(var_preds).reshape(xx.shape)
cf2 = axs[1].contourf(xx, yy, uncertainty, levels=100, cmap="viridis")
axs[1].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=50)
axs[1].set_title("Predictive Uncertainty (tuned NN)", fontsize=34, pad=15)
axs[1].set_xlabel("x1", fontsize=26)
axs[1].set_ylabel("x2", fontsize=26)
axs[1].tick_params(labelsize=22)
cbar2 = fig.colorbar(cf2, ax=axs[1])
cbar2.ax.tick_params(labelsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Auto-save the plot
plt.savefig("plots/mcd_cla_tuned.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()