### GATE MODEL CODE
### Jake Sumner
### 3/10/25


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GraphAttentionAutoencoder(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, hidden_dim, latent_dim, num_nodes, num_edges, threshold=0.5):
        """
        Args:
            in_node_feats (int): Number of input node features.
            in_edge_feats (int): Number of input edge features.
            hidden_dim (int): Hidden dimension for intermediate layers.
            latent_dim (int): Dimension of the latent embedding Z.
            num_nodes (int): Number of nodes in the graph (assumed fixed).
            num_edges (int): Number of edges in the graph (assumed fixed).
            threshold (float): Threshold to binarize the reconstructed connectivity matrix.
        """
        super(GraphAttentionAutoencoder, self).__init__()
        # Encoder: Two GAT layers.
        self.gat1 = GATConv(in_channels=in_node_feats, out_channels=hidden_dim, heads=1, concat=True)
        self.gat2 = GATConv(in_channels=hidden_dim, out_channels=latent_dim, heads=1, concat=False)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.threshold = threshold

        # Decoder: Three separate heads.
        # Node feature decoder: outputs a vector that is reshaped to (num_nodes, in_node_feats)
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * in_node_feats)
        )
        # Edge feature decoder: outputs a vector that is reshaped to (num_edges, in_edge_feats)
        self.edge_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_edges * in_edge_feats)
        )
        # Connectivity decoder: outputs a vector that is reshaped to (num_nodes, num_nodes)
        # A sigmoid activation is used to constrain the outputs between 0 and 1.
        self.conn_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * num_nodes),
            nn.Sigmoid()
        )
        
    def encode(self, data):
        """
        Encodes the input graph data into a latent vector Z.
        Expects a torch_geometric.data.Data object with attributes:
          - x: node feature matrix of shape [num_nodes, in_node_feats]
          - edge_index: edge list tensor of shape [2, num_edges]
          - batch (optional): batch vector if using multiple graphs in one batch.
        """
        x, edge_index = data.x, data.edge_index
        # If no batch info is provided, assume a single graph.
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        # Global mean pooling to obtain a latent vector Z per graph.
        Z = global_mean_pool(x, batch)  # shape: [num_graphs, latent_dim]
        return Z

    def decode(self, Z):
        """
        Decodes a latent vector (or a batch of latent vectors) Z back into the graph components.
        
        Args:
            Z (torch.Tensor): Latent embedding(s) of shape [latent_dim] or [batch_size, latent_dim].
        
        Returns:
            dict or list: If a single latent vector is provided, returns a dictionary with:
                - node_recon: Reconstructed node feature matrix.
                - edge_recon: Reconstructed edge feature matrix.
                - conn_matrix: Reconstructed connectivity matrix.
                - predicted_edge_list: Edge list derived from thresholding the connectivity matrix.
            If a batch is provided, returns a list of such dictionaries.
        """
        if Z.dim() == 1:
            # Decode a single graph's latent vector
            node_recon = self.node_decoder(Z).view(self.num_nodes, -1)
            edge_recon = self.edge_decoder(Z).view(self.num_edges, -1)
            conn_matrix = self.conn_decoder(Z).view(self.num_nodes, self.num_nodes)
            predicted_edge_list = (conn_matrix > self.threshold).nonzero(as_tuple=False)
            return {
                "node_recon": node_recon,
                "edge_recon": edge_recon,
                "conn_matrix": conn_matrix,
                "predicted_edge_list": predicted_edge_list
            }
        elif Z.dim() == 2:
            # Decode a batch of latent vectors
            decoded_list = []
            for z in Z:
                node_recon = self.node_decoder(z).view(self.num_nodes, -1)
                edge_recon = self.edge_decoder(z).view(self.num_edges, -1)
                conn_matrix = self.conn_decoder(z).view(self.num_nodes, self.num_nodes)
                predicted_edge_list = (conn_matrix > self.threshold).nonzero(as_tuple=False)
                decoded_list.append({
                    "node_recon": node_recon,
                    "edge_recon": edge_recon,
                    "conn_matrix": conn_matrix,
                    "predicted_edge_list": predicted_edge_list
                })
            return decoded_list
        else:
            raise ValueError("Latent vector Z must be 1D or 2D.")

    def forward(self, data):
        """
        Forward pass that first encodes the graph data into Z and then decodes Z back into graph components.
        """
        Z = self.encode(data)
        decoded = self.decode(Z)
        if isinstance(decoded, list):
            return {"decoded": decoded, "Z": Z}
        else:
            return {**decoded, "Z": Z}


class PredictiveModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        """
        Predictive model that takes the latent vector Z as input and predicts one output feature.
        It uses two linear layers with a ReLU activation between them, and applies softmax at the final layer.
        
        Args:
            latent_dim (int): Dimension of the input latent vector.
            hidden_dim (int): Hidden dimension for the intermediate layer.
        """
        super(PredictiveModel, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # output dimension is 1
      

    def forward(self, z):
        """
        Args:
            z (torch.Tensor): The latent vector (shape: [latent_dim]) or a batch of latent vectors.
        Returns:
            torch.Tensor: Predicted output with softmax applied.
        """
        x = F.relu(self.fc1(z))
        x = self.fc2(x)
        return x


# === Example Usage ===
if __name__ == '__main__':
    from torch_geometric.data import Data

    # Suppose we have a graph with 5 nodes and 8 edges.
    num_nodes = 5
    num_edges = 8
    in_node_feats = 10
    in_edge_feats = 3
    hidden_dim = 16
    latent_dim = 8

    # Dummy data (for a single graph)
    x = torch.randn((num_nodes, in_node_feats))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn((num_edges, in_edge_feats))
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Instantiate the autoencoder model
    model = GraphAttentionAutoencoder(
        in_node_feats=in_node_feats,
        in_edge_feats=in_edge_feats,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_nodes=num_nodes,
        num_edges=num_edges,
        threshold=0.5
    )
    
    # Forward pass: encode and decode
    output = model(data)
    
    print("Reconstructed node features:")
    print(output["node_recon"])
    print("\nReconstructed edge features:")
    print(output["edge_recon"])
    print("\nReconstructed connectivity matrix:")
    print(output["conn_matrix"])
    print("\nPredicted edge list (indices where connectivity > threshold):")
    print(output["predicted_edge_list"])
    print("\nLatent vector Z:")
    print(output["Z"])
    
    # Decoding directly from a latent vector Z
    Z = output["Z"][0] if output["Z"].dim() > 1 else output["Z"]
    decoded_output = model.decode(Z)
    print("\nDecoded output from provided latent vector Z:")
    print(decoded_output)
