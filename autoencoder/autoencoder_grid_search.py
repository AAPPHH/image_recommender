import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# --- VLAD-Vektoren laden ---
with open("vlad_vectors_20000.pkl", "rb") as f:
    data = pickle.load(f)
vectors = data["vectors"].astype(np.float32)
paths = data["paths"]

# --- Feste Testmenge: z.B. 50 zufällige, aber immer gleich durch seed ---
random.seed(42)
n_test = 500
idx_all = list(range(len(vectors)))
test_idx = random.sample(idx_all, n_test)
train_idx = list(set(idx_all) - set(test_idx))
X_test = vectors[test_idx]
X_train = vectors[train_idx]
input_dim = vectors.shape[1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --- Nur Encoder (kein Decoder mehr) ---
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, dropout_rate=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, p=2, dim=-1)
        return z

def isometry_loss_corr(x, z, sample_k=None, eps=1e-8):
    """
    Computes a correlation-based isometry loss that penalizes mismatch in relative distances.

    Args:
        x (torch.Tensor): Original input vectors.
        z (torch.Tensor): Latent representations.
        sample_k (int, optional): If set, randomly subsamples sample_k entries for efficiency.
        eps (float): Numerical stability epsilon.

    Returns:
        torch.Tensor: Loss value (1 - Pearson correlation coefficient).
    """
    if sample_k is not None and sample_k < x.size(0):
        idx = torch.randperm(x.size(0), device=x.device)[:sample_k]
        x, z = x[idx], z[idx]
    D_x = torch.cdist(x, x)
    D_z = torch.cdist(z, z)
    Dx_flat = D_x.triu(1).flatten()
    Dz_flat = D_z.triu(1).flatten()
    Dx_mean = Dx_flat.mean()
    Dz_mean = Dz_flat.mean()
    Dx_centered = Dx_flat - Dx_mean
    Dz_centered = Dz_flat - Dz_mean
    corr_num = (Dx_centered * Dz_centered).sum()
    corr_den = (Dx_centered.pow(2).sum().sqrt() * Dz_centered.pow(2).sum().sqrt()) + eps
    corr = corr_num / corr_den
    return 1 - corr

def umap_loss(x, z, temperature=1.0):
    """
    Computes a KL divergence between pairwise distance softmaxes in original and latent space.

    Args:
        x (torch.Tensor): Original vectors.
        z (torch.Tensor): Latent vectors.
        temperature (float): Scaling factor for pairwise distances.

    Returns:
        torch.Tensor: KL divergence loss.
    """
    D_x = torch.cdist(x, x)
    D_z = torch.cdist(z, z)
    probs_x = torch.softmax(-D_x / temperature, dim=1)
    probs_z = torch.softmax(-D_z / temperature, dim=1)
    return F.kl_div(probs_z.log(), probs_x, reduction="batchmean")

def train_encoder_corr_umap(
    X_train, input_dim, latent_dim=64, dropout_rate=0.1,
    lambda_corr=1.0, lambda_umap=0.1, temperature=1.0,
    epochs=30, lr=1e-3, batch_size=64, device='cpu', sample_k=64
):
    """
    Trains the encoder using a combination of isometry loss and UMAP-style KL loss.

    Args:
        X_train (np.ndarray): Training data array of shape (n_samples, input_dim).
        input_dim (int): Input vector dimensionality.
        latent_dim (int): Latent vector dimensionality.
        dropout_rate (float): Dropout probability.
        lambda_corr (float): Weight for the isometry loss.
        lambda_umap (float): Weight for the UMAP loss.
        temperature (float): Softmax temperature for UMAP loss.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size for training.
        device (str): Device to use ("cpu" or "cuda").
        sample_k (int): Number of samples to use in pairwise distance calculations.

    Returns:
        Encoder: Trained encoder model in evaluation mode.
    """
    model = Encoder(input_dim=input_dim, latent_dim=latent_dim, dropout_rate=dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    dataset = TensorDataset(torch.from_numpy(X_train).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    for epoch in range(epochs):
        losses = []
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model(batch)
            loss_corr = isometry_loss_corr(batch, z, sample_k=sample_k)
            loss_umap = umap_loss(batch, z, temperature=temperature)
            loss = lambda_corr * loss_corr + lambda_umap * loss_umap
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: loss={np.mean(losses):.6f}, corr={loss_corr.item():.5f}, umap={loss_umap.item():.5f}")
    model.eval()
    return model

def neighbor_preservation(X_orig, X_lat, k=100):
    """
    Evaluates neighborhood preservation between original and latent spaces.

    Args:
        X_orig (np.ndarray): Original feature vectors.
        X_lat (np.ndarray): Latent feature vectors.
        k (int): Number of nearest neighbors to consider.

    Returns:
        float: Average overlap ratio of top-k neighbors.
    """
    nn_orig = NearestNeighbors(n_neighbors=k+1).fit(X_orig)
    nn_lat = NearestNeighbors(n_neighbors=k+1).fit(X_lat)
    idx_orig = nn_orig.kneighbors(X_orig, return_distance=False)[:, 1:]
    idx_lat = nn_lat.kneighbors(X_lat, return_distance=False)[:, 1:]

    overlap = [
        len(set(orig).intersection(set(lat))) / k
        for orig, lat in zip(idx_orig, idx_lat)
    ]
    return np.mean(overlap)

n_samples = X_test.shape[0]
orig_dists = squareform(pdist(X_test, metric="euclidean"))
orig_flat = orig_dists[np.triu_indices(n_samples, k=1)]

latent_dims = [128]
dropouts = [0.1]
lambda_corrs = [2]
lambda_umaps = [0.25]
temperatures = [1.5]

results = []

for latent_dim in latent_dims:
    for dropout in dropouts:
        for lambda_corr in lambda_corrs:
            for lambda_umap in lambda_umaps:
                for temperature in temperatures:
                    print(f"\n=== latent_dim={latent_dim} dropout={dropout} lambda_corr={lambda_corr} lambda_umap={lambda_umap} temperature={temperature} ===")
                    model = train_encoder_corr_umap(
                        X_train, input_dim, latent_dim=latent_dim,
                        dropout_rate=dropout,
                        lambda_corr=lambda_corr,
                        lambda_umap=lambda_umap,
                        temperature=temperature,
                        epochs=30, batch_size=512, device=device, sample_k=256
                    )

                    with torch.no_grad():
                        X_test_tensor = torch.from_numpy(X_test).float().to(device)
                        latents = model(X_test_tensor)
                        latents = latents.cpu().numpy()
                    latent_dists = squareform(pdist(latents, metric="euclidean"))
                    latent_flat = latent_dists[np.triu_indices(n_test, k=1)]
                    corr = pearsonr(orig_flat, latent_flat)[0]
                    mse = np.mean((orig_flat - latent_flat) ** 2)
                    print(f"  -> Corr: {corr:.4f}, MSE: {mse:.6f}")
                    score = neighbor_preservation(X_test, latents, k=10)
                    print(f"Mean Top-10 Neighborhood Overlap: {score:.4f}")

                    results.append({
                        "latent_dim": latent_dim,
                        "dropout": dropout,
                        "lambda_corr": lambda_corr,
                        "lambda_umap": lambda_umap,
                        "temperature": temperature,
                        "corr": corr,
                        "mse": mse,
                        "np@10": score
                    })

# --- Ergebnisse als DataFrame ---
df = pd.DataFrame(results)
print("\nTop-Setups nach Pearson-Korrelation:")
print(df.sort_values("corr", ascending=False).head(10))
print("\nTop-Setups nach MSE:")
print(df.sort_values("mse", ascending=True).head(10))
