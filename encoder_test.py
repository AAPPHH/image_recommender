import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import h5py
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

# -------------------------------------
# Pfade
# -------------------------------------
ENCODER_PATH = "sift_vlad_encoder.pt"
H5_PATH = "vlad_vectors.hdf5"
n_test = 500
n_load = 20000   # Wie viele Vektoren du insgesamt laden willst

# -------------------------------------
# Dataset Klasse für HDF5-Zugriff
# -------------------------------------
class H5VLADDataset:
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

# -------------------------------------
# Test-/Train-Split mit festem Seed
# -------------------------------------
with h5py.File(H5_PATH, "r") as f:
    N = min(n_load, f["vectors"].shape[0])
    input_dim = f["vectors"].shape[1]

random.seed(42)
idx_all = list(range(N))
test_idx = random.sample(idx_all, n_test)
train_idx = list(set(idx_all) - set(test_idx))

# -------------------------------------
# Testdaten laden (wie bisher: als Array)
# -------------------------------------
# Damit du den Batch später als Array hast:
test_dataset = H5VLADDataset(H5_PATH, test_idx)
X_test = np.stack([test_dataset[i] for i in range(len(test_dataset))])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Test: {X_test.shape}")

# -------------------------------------
# Encoder-Definition (wie beim Training!)
# -------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim=32768, latent_dim=128, dropout_rate=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 669),
            nn.LayerNorm(669),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(669, 317),
            nn.LayerNorm(317),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(317, latent_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, p=2, dim=-1)
        return z

latent_dim = 128 # Passe ggf. an!
dropout_rate = 0.1

encoder = Encoder(input_dim, latent_dim=latent_dim, dropout_rate=dropout_rate).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.eval()

# -------------------------------------
# In den Latent Space projizieren
# -------------------------------------
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    latents = encoder(X_test_tensor)
    latents = latents.cpu().numpy()

# -------------------------------------
# Korrelation der Distanzmatrizen
# -------------------------------------
orig_dists = squareform(pdist(X_test, metric="euclidean"))
latent_dists = squareform(pdist(latents, metric="euclidean"))
orig_flat = orig_dists[np.triu_indices(n_test, k=1)]
latent_flat = latent_dists[np.triu_indices(n_test, k=1)]

pearson_corr = pearsonr(orig_flat, latent_flat)[0]
mse = np.mean((orig_flat - latent_flat) ** 2)

print(f"Pearson correlation: {pearson_corr:.4f}")
print(f"MSE: {mse:.6f}")
# Test: (500, 32768)
# Pearson correlation: 0.8452
# MSE: 0.124418

# Test: (500, 32768)
# Pearson correlation: 0.8596
# MSE: 0.105387