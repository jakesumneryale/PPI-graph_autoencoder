### VGAE Model ###
### Jake Sumner
### 3/10/25

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class VariationalGraphAutoencoder(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, hidden_dim, latent_dim):
        """
        Args:
            in_node_feats (int): Dimensionality of input node features.
            in_edge_feats (int): Dimensionality of input edge features.
            hidden_dim (int): Hidden dimension used in the GAT layers and decoders.
            latent_dim (int): Dimensionality of the latent space (graph-level embedding).
        """
        super(VariationalGraphAutoencoder, self).__init__()
        # Encoder: two GAT layers to obtain node embeddings.
        self.gat1 = GATConv(in_channels=in_node_feats, out_channels=hidden_dim, heads=1, concat=True)
        self.gat2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=1, concat=False)
        
        # Global pooling: aggregate node embeddings into a graph-level embedding.
        # Variational parameters: compute μ and log(σ²) from the pooled features.
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: generates per-node hidden representations from the global latent vector.
        # (This MLP is applied per graph: we tile the graph’s z to match its node count.)
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # produces a node hidden representation
        )
        # Predict node features from the per-node hidden representation.
        self.node_feature_predictor = nn.Linear(hidden_dim, in_node_feats)
        # Connectivity is obtained via the inner product of node hidden representations.
        # Edge decoder: given a pair of node hidden representations, predict edge features.
        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_edge_feats)
        )
    
    def encode(self, data):
        """
        Encodes a batch of graphs.
        
        Args:
            data (torch_geometric.data.Data): A batch object containing attributes:
                - x: [total_num_nodes, in_node_feats]
                - edge_index: [2, total_num_edges]
                - batch: [total_num_nodes] (mapping nodes to their respective graph)
        Returns:
            mu, logvar: Tensors of shape [num_graphs, latent_dim]
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        # Global mean pooling: one embedding per graph.
        pooled = global_mean_pool(x, batch)  # shape: [num_graphs, hidden_dim]
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Applies the reparameterization trick to sample z."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, data):
        """
        Decodes each graph’s latent vector into node features, a connectivity matrix, and edge features.
        
        Args:
            z (torch.Tensor): Latent vectors for each graph, shape [num_graphs, latent_dim].
            data (torch_geometric.data.Data): The original batch with attributes:
                - batch: node-to-graph mapping.
                - edge_index: global edge indices.
        Returns:
            List of dictionaries (one per graph) with keys:
                - "node_recon": [n, in_node_feats] reconstructed node features.
                - "conn_matrix": [n, n] reconstructed connectivity matrix.
                - "predicted_edge_list": Indices (i, j) where connectivity > 0.5.
                - "edge_recon": [num_edges, in_edge_feats] reconstructed edge features.
        """
        decoded_graphs = []
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        # PyG’s Batch object provides 'ptr' (cumulative node counts) when using DataLoader.
        # If not available, compute it from batch.
        if hasattr(data, 'ptr'):
            ptr = data.ptr
        else:
            counts = torch.bincount(batch)
            ptr = torch.cat([torch.tensor([0], device=z.device), torch.cumsum(counts, dim=0)])
        
        # Process each graph in the batch individually.
        for i in range(z.size(0)):
            # Number of nodes in graph i.
            n = int(ptr[i+1] - ptr[i])
            # Decode node hidden representations: tile z[i] for each node.
            z_i = z[i]  # shape: [latent_dim]
            node_input = z_i.unsqueeze(0).repeat(n, 1)  # shape: [n, latent_dim]
            node_hidden = self.node_decoder(node_input)  # shape: [n, hidden_dim]
            # Reconstruct node features.
            node_recon = self.node_feature_predictor(node_hidden)  # shape: [n, in_node_feats]
            # Compute connectivity matrix via inner product (and apply sigmoid).
            conn_matrix = torch.sigmoid(torch.matmul(node_hidden, node_hidden.t()))
            
            # Filter edge_index for graph i.
            # Get global node indices belonging to graph i.
            node_indices = (batch == i).nonzero(as_tuple=False).view(-1)
            # Select edges whose both endpoints belong to this graph.
            edge_mask = torch.isin(data.edge_index[0], node_indices) & torch.isin(data.edge_index[1], node_indices)
            edge_index_i = data.edge_index[:, edge_mask]
            # Remap global node indices to local indices (0 to n-1).
            mapping = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(node_indices)}
            edge_index_local = edge_index_i.clone()
            for j in range(edge_index_local.size(1)):
                edge_index_local[0, j] = mapping[edge_index_i[0, j].item()]
                edge_index_local[1, j] = mapping[edge_index_i[1, j].item()]
            # Decode edge features for each edge (if any).
            if edge_index_local.size(1) > 0:
                edge_input = torch.cat([node_hidden[edge_index_local[0]], node_hidden[edge_index_local[1]]], dim=1)
                edge_recon = self.edge_decoder(edge_input)  # shape: [num_edges, in_edge_feats]
            else:
                edge_recon = torch.empty((0, self.edge_decoder[-1].out_features), device=z.device)
            
            decoded_graphs.append({
                "node_recon": node_recon,
                "conn_matrix": conn_matrix,
                "predicted_edge_list": (conn_matrix > 0.5).nonzero(as_tuple=False),
                "edge_recon": edge_recon
            })
        return decoded_graphs

    def forward(self, data):
        """
        Processes a batch of graphs.
        
        Returns:
            mu, logvar: Variational parameters for each graph.
            z: Sampled latent vectors, one per graph.
            decoded: List of decoded graph reconstructions.
        """
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, data)
        return mu, logvar, z, decoded

class PredictiveModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        """
        Predictive model that takes a graph’s latent vector z as input and predicts a scalar value.
        
        Args:
            latent_dim (int): Dimensionality of the latent vector.
            hidden_dim (int): Hidden dimension used in the predictor.
        """
        super(PredictiveModel, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = self.fc2(x)
        return x
