# Uncertainty estimation with MC-Dropout and SWAG on Boston Housing dataset

# Ethical issue w/ Boston Housing: Variable 'B' (1000(Bk - 0.63)^2, where Bk is the proportion of Black people by town.
# This variable was engineered under the assumption that racial self-segregation positively impacts house prices, which is problematic and not substantiated
# -> I removed this variable from the dataset

# Packages needed: numpy, pandas, torch, sklearn (scikit-learn), matplotlib, skorch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split

# Set device & cores
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(torch.get_num_threads())

# Set seeds
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)

##########################################################################################
#
# SWAG implementation
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
# 1. Data prep
#
##########################################################################################

def load_boston():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return np.delete(data, 11, axis=1), target  # Remove problematic variable

X, y = load_boston() # shape is now 506, 12 instead of 506, 13

# Split and scale
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=7)

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

X_train = x_scaler.transform(X_train).astype(np.float32)
X_val = x_scaler.transform(X_val).astype(np.float32)
X_test = x_scaler.transform(X_test).astype(np.float32)
y_train = y_scaler.transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)
y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)


##########################################################################################
#
# 2. Models
# 
##########################################################################################

class MCMLP(nn.Module):
    def __init__(self, in_dim=12, h1=128, h2=64, h3=32, h4=16, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h3, h4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h4, 1)
        )
    def forward(self, x): return self.net(x)

class PlainMLP(nn.Module):
    def __init__(self, in_dim=12, h1=128, h2=64, h3=32, h4=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, h4),
            nn.ReLU(),
            nn.Linear(h4, 1)
        )
    def forward(self, x): return self.net(x)


##########################################################################################
#
# 3. Hyperparameter tuning
#
##########################################################################################

# Create validation dataset
val_ds = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
)

net = NeuralNetRegressor(
    module=MCMLP,
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    max_epochs=300,
    callbacks=[EarlyStopping(patience=15, monitor='valid_loss')],
    train_split=predefined_split(val_ds),
    device=device,
    verbose=0
)

params = {
    'module__h1': [128, 256, 512],
    'module__h2': [64, 128, 256],
    'module__h3': [16, 32, 64],
    'module__h4': [8, 16, 32],
    'module__dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'optimizer__lr': [1e-4, 1e-3, 1e2],
    'optimizer__weight_decay': [0, 1e-5, 1e-4, 1e-3]
}

search = RandomizedSearchCV(
    net, params, n_iter=10,
    scoring='neg_mean_squared_error', verbose=1, random_state=7, n_jobs=-1  # Set n_jobs=-1 for parallelization and 1 to avoid it
)

# Fit on training data
search.fit(
    X_train.astype(np.float32), 
    y_train.reshape(-1, 1).astype(np.float32)  # Must be 2D array
)
best_params = search.best_params_
print("Best Params:", best_params)


##########################################################################################
#
# 4. MC-Dropout: Final Training
#
##########################################################################################

mc_model = MCMLP(
    in_dim=12,
    h1=best_params['module__h1'],
    h2=best_params['module__h2'],
    h3=best_params['module__h3'],
    h4=best_params['module__h4'],
    dropout_rate=best_params['module__dropout_rate']
).to(device)

optimizer = optim.Adam(
    mc_model.parameters(), 
    lr=best_params['optimizer__lr'],
    weight_decay=best_params['optimizer__weight_decay']
)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
)
val_ds = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Training loop with early stopping
best_val_loss = float('inf')
patience_counter = 0
patience = 30

for epoch in range(1000):
    # Training
    mc_model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = nn.MSELoss()(mc_model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    
    # Validation
    mc_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = mc_model(xb)
            val_loss += nn.MSELoss()(preds, yb).item() * xb.size(0)
    
    val_loss /= len(val_ds)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(mc_model.state_dict(), 'models/best_mc_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            mc_model.load_state_dict(torch.load('models/best_mc_model.pth'))
            break

# Load best model
mc_model.load_state_dict(torch.load('models/best_mc_model.pth'))

# Combine train+val for final training
print("Training on combined train+val data...")
X_trval = np.vstack([X_train, X_val]).astype(np.float32)
y_trval = np.concatenate([y_train, y_val]).astype(np.float32)
combined_ds = TensorDataset(
    torch.tensor(X_trval, dtype=torch.float32),
    torch.tensor(y_trval, dtype=torch.float32).unsqueeze(1)
)
combined_loader = DataLoader(combined_ds, batch_size=32, shuffle=True)

# Train on combined data
for epoch in range(100):
    mc_model.train()
    epoch_loss = 0.0
    for xb, yb in combined_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = nn.MSELoss()(mc_model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Finetune Epoch {epoch+1}/100: Loss={epoch_loss/len(combined_loader):.4f}")


##########################################################################################
#
# 5. MC-Dropout: Inference
#
##########################################################################################

mc_model.train() # activate dropout
test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

mc_preds = []
with torch.no_grad():
    for _ in range(200):
        mc_preds.append(mc_model(test_tensor).cpu().numpy())
mc_preds = np.concatenate(mc_preds, axis=1)

mc_mean = mc_preds.mean(axis=1).flatten()
mc_std = mc_preds.std(axis=1).flatten()


##########################################################################################
#
# 6. SWAG: Setup
#
##########################################################################################

base_model = PlainMLP(
    in_dim=12,
    h1=best_params['module__h1'],
    h2=best_params['module__h2'],
    h3=best_params['module__h3'],
    h4=best_params['module__h4']
).to(device)

swag = SWAG(base_model, max_rank=20, scale=0.5)
optimizer = optim.Adam(
    base_model.parameters(),
    lr=best_params['optimizer__lr'],
    weight_decay=best_params['optimizer__weight_decay']
)

# Training with burn-in and early stopping
best_val_loss = float('inf')
patience_counter = 0
patience = 30
burn_in = 150

for epoch in range(300): # epochs = 300
    # Training
    base_model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = nn.MSELoss()(base_model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    
    # Validation
    base_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = base_model(xb)
            val_loss += nn.MSELoss()(preds, yb).item() * xb.size(0)
    
    val_loss /= len(val_ds)
    
    # SWAG update after burn-in
    if epoch >= burn_in:
        swag.update()
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(base_model.state_dict(), 'models/best_swag_base.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience and epoch >= burn_in:
            print(f"SWAG early stopping at epoch {epoch}")
            break

# Load best base model
base_model.load_state_dict(torch.load('models/best_swag_base.pth'))
swag.current_params = swag._get_flat_params().clone()  # Update stored params

# Train on combined data for SWAG updates
print("SWAG training on combined data...")
for epoch in range(100):  # epochs = 100
    base_model.train()
    for xb, yb in combined_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = nn.MSELoss()(base_model(xb), yb)
        loss.backward()
        optimizer.step()
    
    # Update SWAG
    if epoch >= 50:  # Start updates after initial epochs
        swag.update()


##########################################################################################
#
# 7. SWAG: Inference
#
##########################################################################################

test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
swag_preds = []

for _ in range(200):
    sampled_model = swag.sample()
    sampled_model.eval()
    with torch.no_grad():
        swag_preds.append(sampled_model(test_tensor).cpu().numpy().flatten())
    swag.restore()

swag_preds = np.array(swag_preds)
swag_mean = swag_preds.mean(axis=0)
swag_std = swag_preds.std(axis=0)


##########################################################################################
#
# 8. Evaluation metrics
#
##########################################################################################

def nll(pred_mean, pred_std, target):
    var = np.clip(pred_std**2, 1e-6, None)
    return 0.5 * np.mean((pred_mean - target)**2 / var + np.log(2 * np.pi * var))

def picp(pred_mean, pred_std, target, z=1.96):
    lower = pred_mean - z * pred_std
    upper = pred_mean + z * pred_std
    return np.mean((target >= lower) & (target <= upper))

print("\n" + "="*50)
print("MC-Dropout Results:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, mc_mean)):.4f}")
print(f"NLL: {nll(mc_mean, mc_std, y_test):.4f}")
print(f"PICP: {picp(mc_mean, mc_std, y_test):.4f}")

print("\nSWAG Results:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, swag_mean)):.4f}")
print(f"NLL: {nll(swag_mean, swag_std, y_test):.4f}")
print(f"PICP: {picp(swag_mean, swag_std, y_test):.4f}")
print("="*50 + "\n")


##########################################################################################
#
# 9. Plotting
#
##########################################################################################

# Define consistent colors
mcd_colour = (0, 0, 1)          # blue for MC-Dropout
swag_colour = (1, 0.5, 0)       # orange for SWAG

# Set global font sizes for stand-alone plots
plt.rcParams.update({
    'axes.titlesize': 34,
    'axes.labelsize': 26,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 28
})

# Uncertainty Plot
plt.figure(figsize=(19.2, 10.8))
plt.scatter(y_test, mc_mean, alpha=0.5, label='MC-Dropout', color=mcd_colour, s=100)
plt.errorbar(y_test, mc_mean, yerr=1.96*mc_std, 
             fmt='none', ecolor=mcd_colour + (0.2,), label='_nolegend_', lw=5, capsize=5)
plt.scatter(y_test, swag_mean, marker='s', alpha=0.7, label='SWAG', color=swag_colour, s=100)
plt.errorbar(y_test, swag_mean, yerr=1.96*swag_std,
             fmt='none', ecolor=swag_colour + (0.3,), label='_nolegend_', lw=5, capsize=5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 'k--', alpha=0.7)
plt.xlabel("True Values (Standardized)")
plt.ylabel("Predictions")
plt.title("Predictions with 95% CI", pad=15)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/bh_uncertainty_intervals.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Residual Plot
plt.figure(figsize=(19.2, 10.8))
plt.scatter(mc_mean, y_test - mc_mean, alpha=0.6, label='MC-Dropout', color=mcd_colour, s=100)
plt.scatter(swag_mean, y_test - swag_mean, alpha=0.6, label='SWAG', color=swag_colour, s=100)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Analysis", pad=15)
plt.legend(loc='lower left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/bh_residuals.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Global font sizes for the 2x2 plot
plt.rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14
})

plt.figure(figsize=(19.2, 10.8))

# 1. Uncertainty distribution (top-left)
plt.subplot(221)
plt.hist(mc_std, bins=30, alpha=0.5, label='MC-Dropout', color=mcd_colour)
plt.hist(swag_std, bins=30, alpha=0.5, label='SWAG', color=swag_colour)
plt.xlabel("Prediction Standard Deviation")
plt.ylabel("Frequency")
plt.title("Uncertainty Distribution", pad=10)
plt.legend()
plt.grid(alpha=0.3)

# 2. Residuals vs Uncertainty (top-right)
plt.subplot(222)
plt.scatter(np.abs(y_test - mc_mean), mc_std, alpha=0.6, label='MC-Dropout', color=mcd_colour)
plt.scatter(np.abs(y_test - swag_mean), swag_std, alpha=0.6, label='SWAG', color=swag_colour)
plt.plot([0, max(np.abs(y_test - mc_mean))], [0, max(mc_std)], 'k--', alpha=0.5)
plt.xlabel("Absolute Residual")
plt.ylabel("Prediction Std")
plt.title("Uncertainty vs Error", pad=10)
plt.legend()
plt.grid(alpha=0.3)

# Calibration plots
def plot_calibration(mean: np.ndarray, 
                     std: np.ndarray, 
                     target: np.ndarray, 
                     ax: plt.Axes, 
                     label: str,
                     color=None):
    """Plot calibration of predictions with uncertainty bands"""
    sorted_idx = np.argsort(mean)
    mean_sorted = mean[sorted_idx]
    std_sorted = std[sorted_idx]
    target_sorted = target[sorted_idx]
    
    # Use color if specified
    if color:
        ax.plot(mean_sorted, target_sorted, 'o', alpha=0.3, label=label, color=color)
        ax.fill_between(mean_sorted, 
                       mean_sorted - 1.96*std_sorted, 
                       mean_sorted + 1.96*std_sorted, 
                       alpha=0.2, color=color)
    else:
        ax.plot(mean_sorted, target_sorted, 'o', alpha=0.3, label=label)
        ax.fill_between(mean_sorted, 
                       mean_sorted - 1.96*std_sorted, 
                       mean_sorted + 1.96*std_sorted, 
                       alpha=0.2)
    
    ax.plot([mean.min(), mean.max()], [mean.min(), mean.max()], 'k--')
    ax.set_xlabel("Predicted Mean")
    ax.set_ylabel("True Value")
    ax.set_title(f"Calibration: {label}", pad=10)
    ax.grid(alpha=0.3)
    
    ax.set_xlim(-2.2, 3.2)
    ax.set_ylim(-3.25, 4.25)

# 3. Calibration plot MC-Dropout (bottom-left)
plt.subplot(223)
plot_calibration(mc_mean, mc_std, y_test, plt.gca(), "MC-Dropout", color=mcd_colour)

# 4. Calibration plot SWAG (bottom-right)
plt.subplot(224)
plot_calibration(swag_mean, swag_std, y_test, plt.gca(), "SWAG", color=swag_colour)

plt.tight_layout()
plt.savefig('plots/bh_uncertainty_comp2x2.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


##########################################################################################
#
# 10. OOD Analysis
#
##########################################################################################

# In-domain vs OOD comparison
high_medinc_mask = X_test[:, 0] > 2.0  # Extreme values for MedInc
print(f"Found {high_medinc_mask.sum()} OOD samples (High MedInc)")

if high_medinc_mask.sum() > 0:
    # In-domain metrics
    id_mc_rmse = np.sqrt(mean_squared_error(y_test, mc_mean))
    id_mc_nll = nll(mc_mean, mc_std, y_test)
    id_swag_rmse = np.sqrt(mean_squared_error(y_test, swag_mean))
    id_swag_nll = nll(swag_mean, swag_std, y_test)
    
    # OOD metrics
    ood_mc_rmse = np.sqrt(mean_squared_error(y_test[high_medinc_mask], mc_mean[high_medinc_mask]))
    ood_mc_nll = nll(mc_mean[high_medinc_mask], mc_std[high_medinc_mask], y_test[high_medinc_mask])
    ood_swag_rmse = np.sqrt(mean_squared_error(y_test[high_medinc_mask], swag_mean[high_medinc_mask]))
    ood_swag_nll = nll(swag_mean[high_medinc_mask], swag_std[high_medinc_mask], y_test[high_medinc_mask])
    
    print("\n" + "="*50)
    print("Natural OOD Analysis (High MedInc):")
    print(f"MC-Dropout RMSE: ID {id_mc_rmse:.4f} vs OOD {ood_mc_rmse:.4f}")
    print(f"MC-Dropout NLL: ID {id_mc_nll:.4f} vs OOD {ood_mc_nll:.4f}")
    print(f"SWAG RMSE: ID {id_swag_rmse:.4f} vs OOD {ood_swag_rmse:.4f}")
    print(f"SWAG NLL: ID {id_swag_nll:.4f} vs OOD {ood_swag_nll:.4f}")
    print("="*50 + "\n")

# Artificial OOD samples
X_ood = X_test.copy()
high_income_mask = X_ood[:, 0] > 1.5  # Above average income
X_ood[high_income_mask, 0] *= 3.0  # Exaggerate high income
X_ood[high_income_mask, 5] *= 0.3  # Reduce occupancy

# Evaluate on OOD samples
test_tensor_ood = torch.tensor(X_ood, dtype=torch.float32).to(device)

# MC-Dropout OOD
mc_model.train()
mc_preds_ood = []
with torch.no_grad():
    for _ in range(200):
        mc_preds_ood.append(mc_model(test_tensor_ood).cpu().numpy())
mc_preds_ood = np.concatenate(mc_preds_ood, axis=1)
mc_mean_ood = mc_preds_ood.mean(axis=1).flatten()
mc_std_ood = mc_preds_ood.std(axis=1).flatten()

# SWAG OOD
swag_preds_ood = []
for _ in range(200):
    sampled_model = swag.sample()
    sampled_model.eval()
    with torch.no_grad():
        swag_preds_ood.append(sampled_model(test_tensor_ood).cpu().numpy().flatten())
    swag.restore()
swag_preds_ood = np.array(swag_preds_ood)
swag_mean_ood = swag_preds_ood.mean(axis=0)
swag_std_ood = swag_preds_ood.std(axis=0)

# Compute in-domain metrics (for comparison)
id_mc_rmse = np.sqrt(mean_squared_error(y_test, mc_mean))
id_mc_nll = nll(mc_mean, mc_std, y_test)
id_swag_rmse = np.sqrt(mean_squared_error(y_test, swag_mean))
id_swag_nll = nll(swag_mean, swag_std, y_test)

# Compute artificial OOD metrics
ood_mc_rmse_art = np.sqrt(mean_squared_error(y_test, mc_mean_ood))
ood_mc_nll_art = nll(mc_mean_ood, mc_std_ood, y_test)
ood_swag_rmse_art = np.sqrt(mean_squared_error(y_test, swag_mean_ood))
ood_swag_nll_art = nll(swag_mean_ood, swag_std_ood, y_test)

print("\n" + "="*50)
print("Artificial OOD Results:")
print(f"MC-Dropout RMSE: ID {id_mc_rmse:.4f} vs OOD {ood_mc_rmse_art:.4f}")
print(f"MC-Dropout NLL: ID {id_mc_nll:.4f} vs OOD {ood_mc_nll_art:.4f}")
print(f"SWAG RMSE: ID {id_swag_rmse:.4f} vs OOD {ood_swag_rmse_art:.4f}")
print(f"SWAG NLL: ID {id_swag_nll:.4f} vs OOD {ood_swag_nll_art:.4f}")
print(f"MC-Dropout avg std: ID {mc_std.mean():.4f} vs OOD {mc_std_ood.mean():.4f}")
print(f"SWAG avg std: ID {swag_std.mean():.4f} vs OOD {swag_std_ood.mean():.4f}")
print("="*50)

# Plot OOD comparison
plt.figure(figsize=(19.2, 10.8))

# MC-Dropout OOD comparison
plt.subplot(121)
plt.hist(mc_std, bins=30, alpha=0.7, label='In-domain', color='grey')
plt.hist(mc_std_ood, bins=30, alpha=0.5, label='OOD', color='orange')
plt.xlabel("MC-Dropout Std", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.title("MC-Dropout Uncertainty: ID vs OOD", fontsize=32, pad=10)
plt.legend(fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0, 1)
plt.ylim(0, 21)
plt.grid(alpha=0.3)

# SWAG OOD comparison
plt.subplot(122)
plt.hist(swag_std, bins=30, alpha=0.7, label='In-domain', color='grey')
plt.hist(swag_std_ood, bins=30, alpha=0.5, label='OOD', color='orange')
plt.xlabel("SWAG Std", fontsize=20)
plt.title("SWAG Uncertainty: ID vs OOD", fontsize=32, pad=10)
plt.legend(fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0, 1)
plt.ylim(0, 21)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/bh_ood_comp.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
