import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.style.use('dark_background')
BG_COLOR = "#0a0a0f"
TEXT_COLOR = "white"

def setup_figure(figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    return fig, ax

os.makedirs("image", exist_ok=True)

print("Cargando dataset real...")
ruta_csv = r"C:\Universidad\Machine Learning\lab01\data\UJIndoorLoc\trainingData.csv"
df = pd.read_csv(ruta_csv)

X = df.loc[:, 'WAP001':'WAP520'].copy()
X[X == 100] = -105
y = df['BUILDINGID'].astype(str) + "_" + df['FLOOR'].astype(str)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

muestra_idx = np.random.choice(len(X_scaled), 4000, replace=False)
X_muestra = X_scaled[muestra_idx]
y_muestra = y.iloc[muestra_idx]

print("Calculando PCA y t-SNE...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_muestra)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_muestra)

y_numerico = y_muestra.astype('category').cat.codes

fig, ax = setup_figure()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numerico, cmap='plasma', alpha=0.7, s=15)
plt.title("Proyección PCA (Datos Reales)", color=TEXT_COLOR, fontsize=16)
plt.savefig("image/pca.png", dpi=300, bbox_inches='tight')
plt.close()

fig, ax = setup_figure()
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_numerico, cmap='plasma', alpha=0.7, s=15)
plt.title("Manifold t-SNE (Datos Reales)", color=TEXT_COLOR, fontsize=16)
plt.savefig("image/tsne.png", dpi=300, bbox_inches='tight')
plt.close()

print("Ejecutando Clustering en datos reales...")
kmeans = KMeans(n_clusters=13, random_state=42, n_init=10)
y_km = kmeans.fit_predict(X_pca)

dbscan = DBSCAN(eps=0.8, min_samples=15)
y_db = dbscan.fit_predict(X_pca)

agg = AgglomerativeClustering(n_clusters=13)
y_agg = agg.fit_predict(X_pca)

modelos_cluster = {
    "kmeans": y_km,
    "dbscan": y_db,
    "agg": y_agg
}

for nombre, preds in modelos_cluster.items():
    fig, ax = setup_figure()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=preds, cmap='viridis', alpha=0.7, s=15)
    plt.title(f"Clustering: {nombre.upper()}", color=TEXT_COLOR, fontsize=16)
    plt.savefig(f"image/{nombre}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("Calculando Métricas...")
for nombre, preds in modelos_cluster.items():
    try:
        sil = silhouette_score(X_pca, preds)
    except:
        sil = 0.0

    ari = adjusted_rand_score(y_muestra, preds)

    fig, ax = setup_figure(figsize=(5, 4))
    barras = ax.bar(["Silhouette", "ARI"], [sil, max(0, ari)], color=['#00ff00', '#ff0000'], alpha=0.8)
    ax.set_ylim(0, max(1, sil + 0.2))
    ax.set_title(f"Métricas Reales: {nombre.upper()}", color=TEXT_COLOR)

    for barra in barras:
        yval = barra.get_height()
        ax.text(barra.get_x() + barra.get_width() / 2, yval + 0.02, round(yval, 3),
                ha='center', va='bottom', color=TEXT_COLOR, fontweight='bold')

    plt.savefig(f"image/res_{nombre}.png", dpi=300, bbox_inches='tight')
    plt.close()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

print("Iniciando Deep Learning con PyTorch...")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_data = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(520, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 520)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class ClasificadorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 13)
        )

    def forward(self, x):
        return self.net(x)

autoencoder = Autoencoder()
mlp = ClasificadorMLP()

criterion_ae = nn.MSELoss()
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=0.001)

criterion_mlp = nn.CrossEntropyLoss()
optimizer_mlp = optim.Adam(mlp.parameters(), lr=0.001)

print("Entrenando Autoencoder...")
for epoch in range(50):
    for batch_x, _ in train_loader:
        optimizer_ae.zero_grad()
        _, decoded = autoencoder(batch_x)
        loss = criterion_ae(decoded, batch_x)
        loss.backward()
        optimizer_ae.step()

print("Entrenando MLP...")
with torch.no_grad():
    bottleneck_train, _ = autoencoder(X_train_t)
    bottleneck_test, _ = autoencoder(X_test_t)

dataset_mlp = TensorDataset(bottleneck_train, y_train_t)
loader_mlp = DataLoader(dataset_mlp, batch_size=64, shuffle=True)

for epoch in range(100):
    for batch_x, batch_y in loader_mlp:
        optimizer_mlp.zero_grad()
        outputs = mlp(batch_x)
        loss = criterion_mlp(outputs, batch_y)
        loss.backward()
        optimizer_mlp.step()

print("Generando Matriz de Confusión...")
with torch.no_grad():
    mlp.eval()
    predicciones_raw = mlp(bottleneck_test)
    _, predicciones = torch.max(predicciones_raw, 1)

cm = confusion_matrix(y_test_t.numpy(), predicciones.numpy())

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

sns.heatmap(cm, annot=False, cmap='mako', cbar=True, ax=ax, linewidths=0.5, linecolor=BG_COLOR)
ax.set_title("Matriz de Confusión REAL (Autoencoder + MLP en PyTorch)", color=TEXT_COLOR, fontsize=16, pad=20)
ax.set_xlabel("Predicción", color=TEXT_COLOR, fontsize=14)
ax.set_ylabel("Real", color=TEXT_COLOR, fontsize=14)
ax.tick_params(colors=TEXT_COLOR)

plt.savefig("image/matriz_real.png", dpi=300, bbox_inches='tight')
plt.close()

print("Proceso completado.")
