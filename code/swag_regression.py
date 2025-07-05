# Uncertainty estimation using SWAG on an MLP model for regression.
# Model is trained on a synthetic regression task, and SWAG inference is performed to obtain predictive mean and uncertainty estimates.

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
y = np.sin(x) + 0.3 * np.random.randn(N) # sinus curve + noise
x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=7)
x_test = np.linspace(-8, 8, 100)
y_test = np.sin(x_test) # true function

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)


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
        )
    def forward(self, x):
        return self.net(x)

# Set hyperparameters
epochs = 1000
lr = 1e-3
wd = 1e-5
batch = 20
swag_samples = 200
swag_start = 900  # Start collecting weights after 900 epochs (burn-in is over by then)

# Create DataLoader for training
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch, shuffle=True)


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

# Initialize model
model = MLP(1, 32, 32, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = nn.MSELoss()
swag = SWAG(model, max_rank=50, scale=0.5)

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
    # Update SWAG after burn-in
    if epoch >= swag_start:
        swag.update()

    if (epoch + 1) % 100 == 0:
        print(f"[Epoch {epoch+1}/{epochs}] loss: {epoch_loss/len(train_loader):.4f}")

# SWAG inference on test set
preds = []
with torch.no_grad():
    for _ in range(swag_samples):
        sampled_model = swag.sample()
        preds.append(sampled_model(x_test_tensor).cpu().numpy())
        swag.restore()

preds = np.stack(preds, axis=0)
mean_preds = preds.mean(axis=0).flatten()
std_preds = preds.std(axis=0).flatten()

# Calculate uncertainty metrics
train_range = (x_test >= -5) & (x_test <= 5)
extrap_range = (x_test < -5) | (x_test > 5)

print("\nUncertainty Metrics:")
print(f" In-domain: {std_preds[train_range].mean():.4f}")
print(f" Extrapolation: {std_preds[extrap_range].mean():.4f}")
print(f" Ratio: {std_preds[extrap_range].mean() / std_preds[train_range].mean():.2f}x")

##########################################################################################
#
# 5. Plotting
#
##########################################################################################

plt.style.use('default')
plt.figure(figsize=(19.2, 10.8))

# True function
plt.plot(x_test, np.sin(x_test), linestyle='--', color='black', linewidth=3, label='True function')

# SWAG samples
for i in range(preds.shape[0]):
    plt.scatter(x_test, preds[i,:,0], s=10, alpha=0.5, color='lightblue', label='SWAG samples' if i == 0 else "")

# plot every 5th sample
""" for i in range(0, swag_samples, 5):
    plt.scatter(
        x_test, preds[i, :, 0], 
       s=10, alpha=0.5, color='lightblue',
        label='SWAG Samples' if i == 0 else ""
    ) """

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
    Line2D([0], [0], marker='o', color='lightblue', markersize=10, linestyle='None', label='SWAG samples', alpha=1.0),
    Line2D([0], [0], color='crimson', linewidth=3, label='Mean prediction'),
    Line2D([0], [0], color='crimson', linewidth=10, alpha=0.3, label='Uncertainty (±2 std)')
]

plt.title('SWAG: Uncertainty Estimation on a Regression Task', fontsize=34, pad=15)
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
plt.savefig("plots/swag_regression.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# This code implements SWAG (Stochastic Weight Averaging-Gaussian) for a synthetic regression task:
# - Trains an MLP on noisy sine data
# - Collects model snapshots after a certain epoch (swag_start)
# - At test time samples from these saved models to generate predictions
# - Plots the true function, mean predictions, and uncertainty bands (+- 2 std)


# We see the vertical pillars of sample points again:
# - the test set (x_test) consists of 100 equidistant points in the interval [-8, 8] (every pillar is on points x coordinate)
# - theres structure in the samples, if connected they form smooth lines (connecting the highest points of each pillar)
#   -> This is because these points come from the same set of weights, i.e. the same model 
# - SWAGS uncertainty bands are narrower than MCD, especially in training region
#   -> SWAG is more confident in its predictions




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
y = np.sin(x) + 0.3 * np.random.randn(N)  # sinus curve + noise

# Split into train+validation (80%) and test (20%)
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
# Further split train+validation into train (75% of 80% = 60% of total) and validation (25% of 80% = 20% of total)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=7)

# Create test set
x_test_extrap = np.linspace(-8, 8, 100)
y_test_extrap = np.sin(x_test_extrap)  # true function

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
    def __init__(self, n_in, n_hidden1, n_hidden2, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
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
}

# Create test_fold array: -1 for train, 0 for validation
test_fold = np.array([-1] * len(X_tuning) + [0] * len(X_val))
ps = PredefinedSplit(test_fold)

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
# 4. Final Training (on combined train+validation set)
#
##########################################################################################

# Set hyperparameters from tuning results
epochs = 1000
lr = best_params['lr']
wd = best_params['optimizer__weight_decay']
batch = best_params['batch_size']
n_hidden1 = best_params['module__n_hidden1']
n_hidden2 = best_params['module__n_hidden2']
wd = 1e-5
swag_samples = 200
swag_start = 900  # Start collecting weights after 900 epochs (burn-in is over by then)

# Combine train and validation sets for final training
x_train_full = np.concatenate([x_train, x_val])
y_train_full = np.concatenate([y_train, y_val])
x_train_full_tensor = torch.tensor(x_train_full, dtype=torch.float32).unsqueeze(1)
y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.float32).unsqueeze(1)

# Create DataLoader for training
train_loader = DataLoader(TensorDataset(x_train_full_tensor, y_train_full_tensor), batch_size=batch, shuffle=True)


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

# Initialize model with best hyperparameters
model = MLP(1, n_hidden1, n_hidden2, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = nn.MSELoss()
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
    
    # Update SWAG after burn-in is over
    if epoch >= swag_start:
        swag.update()

    if (epoch + 1) % 100 == 0:
        print(f"[Epoch {epoch+1}/{epochs}] loss: {epoch_loss/len(train_loader):.4f}")


##########################################################################################
#
# 7. Final Testing (on previously unseen test set)
#
##########################################################################################

model.eval()
with torch.no_grad():
    test_pred = model(x_test_tensor)
    test_mse = criterion(test_pred, torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
print(f"\nFinal test MSE (in-distribution): {test_mse.item():.4f}")

# SWAG inference on extrapolation set
preds = []
with torch.no_grad():
    for _ in range(swag_samples):
        sampled_model = swag.sample()
        preds.append(sampled_model(x_test_extrap_tensor).cpu().numpy())
        swag.restore()

preds = np.stack(preds, axis=0)
mean_preds = preds.mean(axis=0).flatten()
std_preds = preds.std(axis=0).flatten()

# Calculate uncertainty metrics
train_range = (x_test_extrap >= -5) & (x_test_extrap <= 5)
extrap_range = (x_test_extrap < -5) | (x_test_extrap > 5)

print("\nUncertainty Metrics:")
print(f" In-domain: {std_preds[train_range].mean():.4f}")
print(f" Extrapolation: {std_preds[extrap_range].mean():.4f}")
print(f" Ratio: {std_preds[extrap_range].mean() / std_preds[train_range].mean():.2f}x")


##########################################################################################
#
# 8. Plotting
#
##########################################################################################

plt.style.use('default')
plt.figure(figsize=(19.2, 10.8))

# True function
plt.plot(x_test_extrap, np.sin(x_test_extrap), linestyle='--', color='black', linewidth=3, label='True function')

# SWAG samples
for i in range(min(100, preds.shape[0])):  # Plot max 100 samples for clarity
    plt.scatter(x_test_extrap, preds[i,:,0], s=10, alpha=0.5, color='lightblue', label='SWAG samples' if i == 0 else "")

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
    Line2D([0], [0], marker='o', color='lightblue', markersize=10, linestyle='None', label='SWAG samples', alpha=1.0),
    Line2D([0], [0], color='crimson', linewidth=3, label='Mean prediction'),
    Line2D([0], [0], color='crimson', linewidth=10, alpha=0.3, label='Uncertainty (±2 std)')
]
plt.title('SWAG: Uncertainty Estimation on a Regression Task (tuned NN)', fontsize=34, pad=15)
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
plt.savefig("plots/swag_reg_tuned.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
