import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

d = 16
sigma = 0.002
lr = 1
epochs = 5000
batch_size = 5128
num_x = 1001
device = "cpu"

class TinyModel(nn.Module):
    def __init__(self, d, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(1001, d)
        self.unembedding = nn.Linear(d, num_classes, bias=False)

    def forward(self, x):
        h = self.embedding(x)
        logits = self.unembedding(h)
        return logits  

model = TinyModel(d, num_x)

optimizer = optim.AdamW(model.parameters())
criterion = nn.CrossEntropyLoss()

def quantize_y(a):
    i = a * (num_x -1)
    i = round(i)
    return max(0, min(num_x - 1, i))

def unquatize_y(a):
    i = a / (num_x - 1)

def f(x, sigma):
    noise = np.random.normal(0, sigma)
    return quantize_y((np.cos(2*np.pi*(x/1000 + noise))+1)/2)


def sample_batch(batch_size):
    x = np.random.randint(0, num_x, size=batch_size)  
    y = np.array([f(val, sigma) for val in x])
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

#def sample_batch(batch_size):
#    x = np.arange(0, batch_size)  
#    indices = np.array([f(val, sigma) for val in x])
#    return torch.tensor(x, dtype=torch.long), torch.tensor(indices, dtype=torch.long)

def plot_embeddings_2d(model, num_points=1000, figsize=(12, 8)):
    model.eval()
    
    x_values = torch.arange(0, num_points, dtype=torch.long)
    
    with torch.no_grad():
        embeddings = model.embedding(x_values).numpy()  # Shape: (num_points, 2)
    
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    colors = x_values.numpy() 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    scatter = ax1.scatter(embeddings[:, 0], embeddings[:, 1], 
                         c=colors, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Original x value', fontsize=12)
    
    ax1.set_xlabel('Embedding dimension 1', fontsize=12)
    ax1.set_ylabel('Embedding dimension 2', fontsize=12)
    ax1.set_title('2D Embeddings Colored by x Value', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(embeddings[:, 0], embeddings[:, 1], 'b-', alpha=0.5, linewidth=1)
    scatter2 = ax2.scatter(embeddings[:, 0], embeddings[:, 1], 
                          c=colors, cmap='viridis', 
                          s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Original x value', fontsize=12)
    
    ax2.set_xlabel('Embedding dimension 1', fontsize=12)
    ax2.set_ylabel('Embedding dimension 2', fontsize=12)
    ax2.set_title('Embedding Trajectory (connected in x order)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Embedding statistics:")
    print(f"  Range dim 1: [{embeddings[:, 0].min():.4f}, {embeddings[:, 0].max():.4f}]")
    print(f"  Range dim 2: [{embeddings[:, 1].min():.4f}, {embeddings[:, 1].max():.4f}]")
    
    return embeddings

def plot_embeddings_3d(model, num_points=1000, figsize=(14, 6),
                           view_angle=(30, 45), show_sphere=True):
    """
    Visualise les embeddings 3D d'un modèle avec deux graphiques côte à côte :
    - Scatter simple
    - Scatter avec trajectoire
    La profondeur est indiquée par l'ombrage automatique de matplotlib (depthshade).
    La barre de couleur est placée à droite, bien distincte des graphiques.
    """
    model.eval()
    x_values = torch.arange(0, num_points, dtype=torch.long)

    with torch.no_grad():
        embeddings = model.embedding(x_values).numpy()

    if embeddings.shape[1] > 3:
        embeddings = embeddings[:, :3]

    # Normalisation L2 (vecteurs de norme 1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Couleurs basées sur la valeur de x
    colors = x_values.numpy()

    # Création des deux sous-graphiques 3D
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   subplot_kw={'projection': '3d'})

    # Configuration commune des axes
    for ax in (ax1, ax2):
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.set_box_aspect([1, 1, 1])  # Égaliser l'échelle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Ajout de la sphère unité (optionnel)
    if show_sphere:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        for ax in (ax1, ax2):
            ax.plot_wireframe(x_sphere, y_sphere, z_sphere,
                              color='gray', alpha=0.1, linewidth=0.5)

    # Graphique 1 : scatter simple (ombrage de profondeur automatique)
    sc = ax1.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                     c=colors, cmap='viridis', s=50, alpha=0.8,
                     edgecolors='black', linewidth=0.5, depthshade=True)
    ax1.set_title('Scatter 3D')

    # Graphique 2 : scatter avec trajectoire
    ax2.plot(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
             color='blue', alpha=0.3, linewidth=1.5)
    ax2.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                c=colors, cmap='viridis', s=50, alpha=0.8,
                edgecolors='black', linewidth=0.5, depthshade=True)
    ax2.set_title('Trajectoire 3D')

    # Ajustement pour laisser de la place à la colorbar à droite
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)

    # Ajout de la colorbar dans un axe dédié
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Valeur de x', fontsize=12)

    plt.show()

    # Statistiques
    print("Statistiques des embeddings (normalisés) :")
    for i in range(3):
        print(f"  Dimension {i+1} : min = {embeddings[:, i].min():.4f}, "
              f"max = {embeddings[:, i].max():.4f}")

    return embeddings

for epoch in range(epochs):
    x, y = sample_batch(batch_size)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

print("Training complete.")

model.eval()
torch.set_printoptions(
    precision=4,          
    threshold=float('inf'),  
    edgeitems=3,           
    linewidth=120,         
    profile='full'         
)

plot_embeddings_2d(model)
plot_embeddings_3d(model)
