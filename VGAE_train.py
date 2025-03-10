### Train VGAE ###
### Jake Sumner
### 3/10/25

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# Example hyperparameters
in_node_feats = 10   # e.g. number of node features
in_edge_feats = 3    # e.g. number of edge features
hidden_dim = 32
latent_dim = 16
batch_size = 32
num_epochs = 50
learning_rate = 0.001

### Define dataset somewhere here ###

# dataset = naomi_data

# Dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Init model
model = VariationalGraphAutoencoder(in_node_feats, in_edge_feats, hidden_dim, latent_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def build_adj_matrix(edge_index, node_indices):
    """
    Helper function to build a ground-truth adjacency matrix for a graph.
    Args:
        edge_index (torch.Tensor): Global edge index of shape [2, num_edges].
        node_indices (torch.Tensor): 1D tensor of global node indices belonging to a graph.
    Returns:
        torch.Tensor: Adjacency matrix of shape [n, n] where n is the number of nodes in the graph.
    """
    n = node_indices.size(0)
    adj = torch.zeros((n, n), device=edge_index.device)
    # Create a mapping from global node index to local index (0, 1, ..., n-1)
    mapping = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(node_indices)}
    # Filter edges that belong to the graph
    mask = (torch.isin(edge_index[0], node_indices) & torch.isin(edge_index[1], node_indices))
    local_edge_index = edge_index[:, mask]
    for j in range(local_edge_index.size(1)):
        u_global = local_edge_index[0, j].item()
        v_global = local_edge_index[1, j].item()
        u_local = mapping[u_global]
        v_local = mapping[v_global]
        adj[u_local, v_local] = 1.0
        # For an undirected graph, make the matrix symmetric
        adj[v_local, u_local] = 1.0
    return adj

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass through the VGAE.
        mu, logvar, z, decoded = model(data)
        batch_loss = 0.0
        
        # data.batch maps each node to its corresponding graph in the batch.
        # decoded is a list of dictionaries (one per graph).
        for i, dec in enumerate(decoded):
            # Obtain the indices of nodes in graph i.
            node_indices = (data.batch == i).nonzero(as_tuple=False).view(-1)
            # Ground-truth node features for graph i.
            x_true = data.x[node_indices]
            # Reconstruction loss for node features.
            loss_node = F.mse_loss(dec["node_recon"], x_true)
            
            # Build the ground-truth adjacency matrix for graph i.
            adj_gt = build_adj_matrix(data.edge_index, node_indices)
            # Reconstruction loss for connectivity.
            loss_conn = F.mse_loss(dec["conn_matrix"], adj_gt)
            
            # Sum reconstruction losses for this graph.
            batch_loss += loss_node + loss_conn
        
        # Average reconstruction loss over the graphs in the batch.
        batch_loss = batch_loss / len(decoded)
        
        # Compute KL divergence loss.
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = batch_loss + kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
