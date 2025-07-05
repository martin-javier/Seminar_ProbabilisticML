# Uncertainty estimation using SWAG on an MLP model for classification.
# Synthetic dataset generated using make_moons from sklearn.
# Model is trained on classification task, SWAG inference is performed to obtain predictive mean and uncertainty estimates.

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

# Generate moon dataset
X, y = make_moons(n_samples=500, noise=0.2)

# Evaluation grid for visualization
xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
X_test_grid = np.c_[xx.ravel(), yy.ravel()]

# Convert to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_grid, dtype=torch.float32)


##########################################################################################
#
# 2. Model
#
##########################################################################################

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_out),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Set hyperparameters
epochs = 1000
lr = 1e-2
wd = 1e-4
batch = 32
swag_samples = 200
swag_start = 900  # Start collecting weights after 900 epochs (burn-in is over by then)

# Create DataLoader for training
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch, shuffle=True)


##########################################################################################
#
# 3. SWAG (https://github.com/wjmaddox/swa_gaussian/)
#
##########################################################################################

class SWAG:
    def __init__(self, model, max_rank=20, var_clamp=1e-5, scale=0.5):
        self.model = model
        self.device = next(model.parameters()).device
        self.base_params = self._get_flat_params().clone()
        self.max_rank = max_rank
        self.var_clamp = var_clamp
        self.scale = scale
        
        # Init SWAG parameters
        self.n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.mean = torch.zeros(self.n_params, device=self.device)
        self.sq_mean = torch.zeros(self.n_params, device=self.device)
        self.deviation_matrix = torch.zeros((self.n_params, max_rank), device=self.device)
        self.n_models = 0
        self.rank = 0
    
    def update(self):
        """Update SWAG parameters with current model weights"""
        params = self._get_flat_params()
        self.n_models += 1
        self.mean = (self.mean * (self.n_models - 1) + params) / self.n_models
        self.sq_mean = (self.sq_mean * (self.n_models - 1) + params**2) / self.n_models
        
        dev = params - self.mean
        if self.rank < self.max_rank:
            self.deviation_matrix[:, self.rank] = dev
            self.rank += 1
        else:
            #self.deviation_matrix = torch.cat((self.deviation_matrix[:, 1:], dev.unsqueeze(1)), dim=1)

            # More efficient: use circular buffer
            self.deviation_matrix = torch.roll(self.deviation_matrix, shifts=-1, dims=1)
            self.deviation_matrix[:, -1] = dev
    
    def sample(self):
        """Sample weights from SWAG posterior"""
        diag_variance = torch.clamp(self.sq_mean - self.mean**2, self.var_clamp)
        z1 = torch.randn(self.n_params, device=self.device) * torch.sqrt(diag_variance)
        z2 = torch.randn(self.rank, device=self.device)
        cov_contribution = self.scale * (self.deviation_matrix[:, :self.rank] @ z2)

        sample = self.mean + z1 + cov_contribution
        self._set_flat_params(sample)
        return self.model
    
    def _get_flat_params(self):
        """Flatten model parameters into a vector"""
        return torch.cat([param.data.view(-1) for param in self.model.parameters()])
    
    def _set_flat_params(self, flat_params):
        """Set model parameters from a flat vector"""
        pointer = 0
        with torch.no_grad():
            for p in self.model.parameters():
                numel = p.numel()
                p.copy_(flat_params[pointer: pointer + numel].view(p.size()))
                pointer += numel
    
    def restore(self):
        """Restore model to its original parameters"""
        self._set_flat_params(self.base_params)


##########################################################################################
#
# 4. Training and testing
#
##########################################################################################

# Initialize model, optimizer, loss
model = MLP(2, 32, 32, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = nn.BCELoss()
swag = SWAG(model, max_rank=20, scale=0.5)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Update SWAG after burn-in
    if epoch >= swag_start:
        swag.update()
    
    if (epoch + 1) % 100 == 0:
        print(f"[Epoch {epoch+1}/{epochs}] loss: {epoch_loss/len(train_loader):.4f}")

# SWAG inference on evaluation grid
preds = []
with torch.no_grad():
    for _ in range(swag_samples):
        sampled_model = swag.sample()
        preds.append(sampled_model(X_test_tensor).cpu().numpy())
        swag.restore()
preds = np.stack(preds, axis=0)

# Compute mean and variance across samples
mean_preds = preds.mean(axis=0)
var_preds = preds.var(axis=0)


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
axs[0].set_title("SWAG: Prediction Probability", fontsize=34, pad=15)
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
plt.savefig("plots/swag_classification.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()




# The same process but with tuning of the MLP hyperparameters using skorch and RandomizedSearchCV:

# comment imports that are done above
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
X_train_val, X_test_holdout, y_train_val, y_test_holdout = train_test_split(X, y, test_size=0.2, random_state=7)

# Further split train+val into train (75%) and validation (25%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=7)

# Evaluation grid for visualization
xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
X_test_grid = np.c_[xx.ravel(), yy.ravel()]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_grid, dtype=torch.float32)
X_test_holdout_tensor = torch.tensor(X_test_holdout, dtype=torch.float32)


##########################################################################################
#
# 2. Model
#
##########################################################################################

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden1, n_hidden2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(1)


##########################################################################################
#
# 3. Hyperparameter Tuning
#
##########################################################################################

# Prepare data for tuning
X_tuning = X_train.astype(np.float32)
y_tuning = y_train.astype(np.float32)
X_val_arr = X_val.astype(np.float32)
y_val_arr = y_val.astype(np.float32)

# Wrap model with skorch
net = NeuralNetClassifier(
    module=MLP,
    module__n_in=2,
    module__n_hidden1=32,
    module__n_hidden2=32,
    max_epochs=300,
    optimizer=torch.optim.Adam,
    optimizer__weight_decay=1e-4,
    criterion=nn.BCEWithLogitsLoss,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    train_split=None,
    verbose=0
)

# Define parameter distribution for randomized search
param_dist = {
    'lr': [1e-3, 1e-2, 5e-2],
    'batch_size': [16, 32, 64],
    'module__n_hidden1': [16, 32, 64],
    'module__n_hidden2': [16, 32, 64],
}

# Predefined split for validation set
test_fold = np.array([-1] * len(X_tuning) + [0] * len(X_val_arr))
ps = PredefinedSplit(test_fold)

# Combine train and validation data
X_combined = np.vstack((X_tuning, X_val_arr))
y_combined = np.concatenate((y_tuning, y_val_arr))

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
# 4. Final Training (on combined train+validation set)
#
##########################################################################################

# Set hyperparameters from tuning results
epochs = 1000
lr = best_params['lr']
batch = best_params['batch_size']
n_hidden1 = best_params['module__n_hidden1']
n_hidden2 = best_params['module__n_hidden2']
wd = 1e-4
swag_samples = 200
swag_start = 900

# Combine train and validation sets for final training
X_train_full = np.concatenate([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])
X_train_full_tensor = torch.tensor(X_train_full, dtype=torch.float32)
y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.float32)

# Create DataLoader for training
train_loader = DataLoader(
    TensorDataset(X_train_full_tensor, y_train_full_tensor), 
    batch_size=batch, 
    shuffle=True
)


##########################################################################################
#
# 5. SWAG (https://github.com/wjmaddox/swa_gaussian/)
#
##########################################################################################

class SWAG:
    def __init__(self, model, max_rank=20, var_clamp=1e-5, scale=0.5):
        self.model = model
        self.device = next(model.parameters()).device
        self.base_params = self._get_flat_params().clone()
        self.max_rank = max_rank
        self.var_clamp = var_clamp
        self.scale = scale
        
        # Init SWAG parameters
        self.n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.mean = torch.zeros(self.n_params, device=self.device)
        self.sq_mean = torch.zeros(self.n_params, device=self.device)
        self.deviation_matrix = torch.zeros((self.n_params, max_rank), device=self.device)
        self.n_models = 0
        self.rank = 0
    
    def update(self):
        """Update SWAG parameters with current model weights"""
        params = self._get_flat_params()
        self.n_models += 1
        self.mean = (self.mean * (self.n_models - 1) + params) / self.n_models
        self.sq_mean = (self.sq_mean * (self.n_models - 1) + params**2) / self.n_models
        
        dev = params - self.mean
        if self.rank < self.max_rank:
            self.deviation_matrix[:, self.rank] = dev
            self.rank += 1
        else:
            #self.deviation_matrix = torch.cat((self.deviation_matrix[:, 1:], dev.unsqueeze(1)), dim=1)

            # More efficient: use circular buffer
            self.deviation_matrix = torch.roll(self.deviation_matrix, shifts=-1, dims=1)
            self.deviation_matrix[:, -1] = dev
    
    def sample(self):
        """Sample weights from SWAG posterior"""
        diag_variance = torch.clamp(self.sq_mean - self.mean**2, self.var_clamp)
        z1 = torch.randn(self.n_params, device=self.device) * torch.sqrt(diag_variance)
        z2 = torch.randn(self.rank, device=self.device)
        cov_contribution = self.scale * (self.deviation_matrix[:, :self.rank] @ z2)

        sample = self.mean + z1 + cov_contribution
        self._set_flat_params(sample)
        return self.model
    
    def _get_flat_params(self):
        """Flatten model parameters into a vector"""
        return torch.cat([param.data.view(-1) for param in self.model.parameters()])
    
    def _set_flat_params(self, flat_params):
        """Set model parameters from a flat vector"""
        pointer = 0
        with torch.no_grad():
            for p in self.model.parameters():
                numel = p.numel()
                p.copy_(flat_params[pointer: pointer + numel].view(p.size()))
                pointer += numel
    
    def restore(self):
        """Restore model to its original parameters"""
        self._set_flat_params(self.base_params)


##########################################################################################
#
# 6. Training and SWAG Collection
#
##########################################################################################

# Initialize model with best hyperparameters - FIXED ARCHITECTURE
model = MLP(2, n_hidden1, n_hidden2)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = nn.BCEWithLogitsLoss()
swag = SWAG(model, max_rank=20, scale=0.5)

# Training with SWAG collection
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Debug check - monitor gradient flow
    if (epoch + 1) % 100 == 0:
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        print(f"[Epoch {epoch+1}/{epochs}] loss: {epoch_loss/len(train_loader):.4f}, avg_grad_norm: {avg_grad_norm:.6f}")
    
    # Update SWAG after burn-in
    if epoch >= swag_start:
        swag.update()

model.eval()
with torch.no_grad():
    logits = model(X_train_full_tensor)
    probs = torch.sigmoid(logits)
    print(f"\nPrediction range: min={probs.min().item():.4f}, max={probs.max().item():.4f}")

# Evaluate on held-out test set
model.eval()
with torch.no_grad():
    logits = model(X_test_holdout_tensor)
    test_preds = torch.sigmoid(logits) > 0.5
    test_acc = (test_preds == torch.tensor(y_test_holdout, dtype=torch.bool)).float().mean()
print(f"\nFinal test accuracy: {test_acc.item():.4f}")

# SWAG inference on evaluation grid
preds = []
with torch.no_grad():
    for _ in range(swag_samples):
        sampled_model = swag.sample()
        logits = sampled_model(X_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
        swag.restore()
preds = np.stack(preds, axis=0)

# Compute mean and variance across samples
mean_preds = preds.mean(axis=0)
var_preds = preds.var(axis=0)


##########################################################################################
#
# 7. Plotting
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
axs[0].set_title("SWAG: Prediction Probability", fontsize=34, pad=15)
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
plt.savefig("plots/swag_cla_tuned.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
