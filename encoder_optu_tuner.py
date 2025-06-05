import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import h5py
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import optuna
from pathlib import Path

BATCH_SIZE = 4096
n_load = 100_000   # Anzahl Samples aus HDF5
n_test = 500     # Größe Testmenge

# ---- HDF5 Dataset-Wrapper für PyTorch ----
class HDF5VectorDataset(Dataset):
    def __init__(self, h5_path, indices):
        self.h5_path = h5_path
        self.indices = np.array(indices)
        with h5py.File(self.h5_path, "r") as f:
            self.vec_len = f["vectors"].shape[1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        with h5py.File(self.h5_path, "r") as f:
            vec = f["vectors"][i][:]
        return vec.astype(np.float32)

# ---- Zufällige Test-/Train-Splits ----
with h5py.File("vlad_vectors.hdf5", "r") as f:
    N = min(n_load, f["vectors"].shape[0])
    input_dim = f["vectors"].shape[1]

random.seed(42)
idx_all = list(range(N))
test_idx = random.sample(idx_all, n_test)
train_idx = list(set(idx_all) - set(test_idx))

# ---- Scipy-pdist Helper für große Matrizen ----
def get_latents(model, dataloader, device):
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            z = model(batch)
            latents.append(z.cpu().numpy())
    return np.vstack(latents)

# ---- Modell mit variabler Architektur ----
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.Mish(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

def isometry_loss_corr(x, z, sample_k=None, eps=1e-8):
    if sample_k is not None and sample_k < x.size(0):
        idx = torch.randperm(x.size(0), device=x.device)[:sample_k]
        x, z = x[idx], z[idx]
    Dx = torch.cdist(x, x)
    Dz = torch.cdist(z, z)
    Dx_f = Dx.triu(1).flatten()
    Dz_f = Dz.triu(1).flatten()
    Dx_c = Dx_f - Dx_f.mean()
    Dz_c = Dz_f - Dz_f.mean()
    corr_num = (Dx_c * Dz_c).sum()
    corr_den = (Dx_c.pow(2).sum().sqrt() * Dz_c.pow(2).sum().sqrt()) + eps
    corr = corr_num / corr_den
    return 1 - corr

def umap_loss(x, z, temperature=1.5):
    Dx = torch.cdist(x, x)
    Dz = torch.cdist(z, z)
    px = torch.softmax(-Dx / temperature, dim=1)
    pz = torch.softmax(-Dz / temperature, dim=1)
    return F.kl_div(pz.log(), px, reduction="batchmean")

def objective(trial):
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 128
    n_layers = trial.suggest_int("n_layers", 1, 3)
    start_size = trial.suggest_int("start_size", 256, 4096, step=64)  # Startgröße des ersten Hidden-Layers
    shrink_ratio = trial.suggest_float("shrink_ratio", 0.4, 0.9)      # Wie stark schrumpfen die Layer?

    dropout = 0.1
    lambda_corr = 2
    lambda_umap = 0.25
    temp = 1.5
    lr = 1e-3
    batch_size = 4096
    epochs = 20

    # Trichterförmig schrumpfende Layer erzeugen:
    hidden_layers = []
    prev_dim = start_size
    for i in range(n_layers):
        next_dim = int(prev_dim * shrink_ratio)
        next_dim = max(next_dim, latent_dim * 2)
        hidden_layers.append(next_dim)
        prev_dim = next_dim

    model = Encoder(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        latent_dim=latent_dim,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_ds = HDF5VectorDataset("vlad_vectors.hdf5", train_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model(batch)
            loss_corr = isometry_loss_corr(batch, z, sample_k=min(256, len(batch)))
            loss_umap = umap_loss(batch, z, temperature=temp)
            loss = lambda_corr * loss_corr + lambda_umap * loss_umap
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        test_ds = HDF5VectorDataset("vlad_vectors.hdf5", test_idx)
        test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=0)
        with torch.no_grad():
            latents = get_latents(model, test_loader, device)
        latent_flat = squareform(pdist(latents))[np.triu_indices(n_test, 1)]

        if epoch == 0:
            global orig_flat
            test_loader_ref = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=0)
            test_data = []
            for batch in test_loader_ref:
                test_data.append(batch.numpy())
            test_data = np.vstack(test_data)
            orig_flat = squareform(pdist(test_data))[np.triu_indices(n_test, 1)]

        corr = pearsonr(orig_flat, latent_flat)[0]
        trial.set_user_attr("hidden_layers", hidden_layers)
        trial.report(corr, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return corr

def print_trial_summary(study, trial):
    if trial.value is not None:
        print(
            f"[Trial {trial.number}] val_corr={trial.value:.4f} | "
            f"n_layers={trial.params.get('n_layers')}, "
            f"start_size={trial.params.get('start_size')}, "
            f"shrink_ratio={trial.params.get('shrink_ratio'):.3f} | "
            f"hidden_layers={trial.user_attrs.get('hidden_layers')}"
        )


if __name__ == "__main__":
    
    study_name = "my_vlad_search"  # Name frei wählbar!
    storage = "sqlite:///optuna_vlad_search.db"  # Relativer Pfad zur DB

    pruner = optuna.pruners.HyperbandPruner(
    min_resource=3,
    max_resource=25,
    reduction_factor=2,
)

    study = optuna.create_study(
        study_name="my_vlad_search",
        direction="maximize",
        storage="sqlite:///optuna_vlad_search.db",
        load_if_exists=True,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=48, callbacks=[print_trial_summary])

    print("🏆 Beste Konfiguration:", study.best_params)
    print("Bestes Hidden-Layer-Setup:", study.best_trial.user_attrs["hidden_layers"])

