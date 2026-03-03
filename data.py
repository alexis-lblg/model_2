import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

d = 3
sigma = 0.004
lr = 1
epochs = 40000
batch_size = 5128
num_x = 1001
device = "cuda"

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
model = model.to(device)

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
    return torch.tensor(x, dtype=torch.long).to(device), torch.tensor(y, dtype=torch.long).to(device)

#def sample_batch(batch_size):
#    x = np.arange(0, batch_size)  
#    indices = np.array([f(val, sigma) for val in x])
#    return torch.tensor(x, dtype=torch.long), torch.tensor(indices, dtype=torch.long)




def plot_embeddings_3d(model, num_points=1000, show_sphere=True):
    model.eval()
    x_values = torch.arange(0, num_points, dtype=torch.long).to(device)

    with torch.no_grad():
        embeddings = model.embedding(x_values).cpu().numpy()
    if embeddings.shape[1] > 3:
        embeddings = embeddings[:, :3]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    colors = x_values.cpu().numpy()

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=('Scatter 3D', 'Trajectory 3D'))

    for col in [1, 2]:
        if show_sphere:
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x_s = np.outer(np.cos(u), np.sin(v)).flatten()
            y_s = np.outer(np.sin(u), np.sin(v)).flatten()
            z_s = np.outer(np.ones(30), np.cos(v)).flatten()
            fig.add_trace(go.Scatter3d(
                x=x_s, y=y_s, z=z_s, mode='markers',
                marker=dict(size=1, color='gray', opacity=0.1), showlegend=False
            ), row=1, col=col)

        if col == 2:
            fig.add_trace(go.Scatter3d(
                x=embeddings[:, 0], y=embeddings[:, 1], z=embeddings[:, 2],
                mode='lines', line=dict(color='blue', width=2), showlegend=False
            ), row=1, col=2)

        fig.add_trace(go.Scatter3d(
            x=embeddings[:, 0], y=embeddings[:, 1], z=embeddings[:, 2],
            mode='markers',
            marker=dict(color=colors, colorscale='Viridis', size=3,
                        showscale=(col == 2), colorbar=dict(title='x value')),
            showlegend=False
        ), row=1, col=col)

    fig.write_html("embeddings_3d.html")
    print("Saved embeddings_3d.html")
    return embeddings


def plot_unembeddings_3d(model, num_points=1000, show_sphere=True):
    model.eval()

    with torch.no_grad():
        unembeddings = model.unembedding.weight.cpu().numpy()[:num_points]
    if unembeddings.shape[1] > 3:
        unembeddings = unembeddings[:, :3]
    unembeddings = unembeddings / np.linalg.norm(unembeddings, axis=1, keepdims=True)
    colors = np.arange(num_points)

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        subplot_titles=('Scatter 3D', 'Trajectory 3D'))

    for col in [1, 2]:
        if show_sphere:
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x_s = np.outer(np.cos(u), np.sin(v)).flatten()
            y_s = np.outer(np.sin(u), np.sin(v)).flatten()
            z_s = np.outer(np.ones(30), np.cos(v)).flatten()
            fig.add_trace(go.Scatter3d(
                x=x_s, y=y_s, z=z_s, mode='markers',
                marker=dict(size=1, color='gray', opacity=0.1), showlegend=False
            ), row=1, col=col)

        if col == 2:
            fig.add_trace(go.Scatter3d(
                x=unembeddings[:, 0], y=unembeddings[:, 1], z=unembeddings[:, 2],
                mode='lines', line=dict(color='blue', width=2), showlegend=False
            ), row=1, col=2)

        fig.add_trace(go.Scatter3d(
            x=unembeddings[:, 0], y=unembeddings[:, 1], z=unembeddings[:, 2],
            mode='markers',
            marker=dict(color=colors, colorscale='Viridis', size=3,
                        showscale=(col == 2), colorbar=dict(title='y value')),
            showlegend=False
        ), row=1, col=col)

    fig.write_html("unembeddings_3d.html")
    print("Saved unembeddings_3d.html")
    return unembeddings

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

plot_embeddings_3d(model)
plot_unembeddings_3d(model)

