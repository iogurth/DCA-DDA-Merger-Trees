import torch
import os
import pickle
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Module, Sequential, ModuleList, Dropout, GELU
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_add_pool, global_max_pool, global_mean_pool



# Deep Decoupled Attention (DDA) Architecture

class LatentGlobalAttention(Module):
    """
    Linear-complexity global graph attention mechanism.
    Projects node features into a latent space to compute attention scores.
    """
    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__()
        self.Wk = Linear(in_channels, out_channels * num_heads)
        self.Wq = Linear(in_channels, out_channels * num_heads)
        self.Wv = Linear(in_channels, out_channels * num_heads)
        self.out_channels = out_channels
        self.num_heads = num_heads

    def forward(self, x):
        qs = self.Wq(x).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(x).reshape(-1, self.num_heads, self.out_channels)
        vs = self.Wv(x).reshape(-1, self.num_heads, self.out_channels)
        
        # L2 Normalization
        qs = qs / torch.norm(qs, p=2, dim=-1, keepdim=True)
        ks = ks / torch.norm(ks, p=2, dim=-1, keepdim=True)
        
        N = qs.shape[0]
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)

        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs) + N * vs

        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)

        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)
        attention_normalizer = torch.unsqueeze(attention_normalizer, -1)
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        
        return (attention_num / attention_normalizer).mean(dim=1)


class DDALocalBlock(Module):
    """
    Local processing block for the DDA architecture.
    Applies GraphSAGE for 1-hop neighborhood aggregation, governed by the 
    residual weight 'alpha', followed by an FFN.
    """
    def __init__(self, hidden_channels, alpha=0.85, dropout=0.1):
        super().__init__()
        self.alpha = alpha
        
        self.norm1 = LayerNorm(hidden_channels)
        self.local_conv = SAGEConv(hidden_channels, hidden_channels)
        
        self.norm2 = LayerNorm(hidden_channels)
        self.ffn = Sequential(
            Linear(hidden_channels, hidden_channels * 2),
            GELU(),
            Dropout(dropout),
            Linear(hidden_channels * 2, hidden_channels)
        )
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index):
        # GNN sub-block with Alpha-controlled residual
        h = self.norm1(x)
        z = F.relu(self.local_conv(h, edge_index))
        x = (1 - self.alpha) * x + self.alpha * self.dropout(z)
        
        # FFN sub-block with direct residual
        h_ffn = self.norm2(x)
        x = x + self.dropout(self.ffn(h_ffn))
        return x


class DDAGlobalBlock(Module):
    """
    Global processing block for the DDA architecture.
    Applies Latent Global Attention across the entire graph, governed by the 
    residual weight 'lmbda', followed by an FFN.
    """
    def __init__(self, hidden_channels, lmbda=0.85, num_heads=4, dropout=0.1):
        super().__init__()
        self.lmbda = lmbda
        
        self.norm1 = LayerNorm(hidden_channels)
        self.global_attn = LatentGlobalAttention(hidden_channels, hidden_channels, num_heads)
        
        self.norm2 = LayerNorm(hidden_channels)
        self.ffn = Sequential(
            Linear(hidden_channels, hidden_channels * 2),
            GELU(),
            Dropout(dropout),
            Linear(hidden_channels * 2, hidden_channels)
        )
        self.dropout = Dropout(dropout)

    def forward(self, x):
        # Attention sub-block with lambda-controlled residual
        h = self.norm1(x)
        z = self.global_attn(h)
        x = (1 - self.lmbda) * x + self.lmbda * self.dropout(z)
        
        # FFN sub-block with direct residual
        h_ffn = self.norm2(x)
        x = x + self.dropout(self.ffn(h_ffn))
        return x


class MLP(Module):
    """
    Multi-Layer Perceptron used for the probabilistic decoding heads.
    """
    def __init__(self, n_in, n_out, hidden=128, nlayers=3):
        super().__init__()
        layers = [Linear(n_in, hidden), GELU()]
        for _ in range(nlayers - 1):
            layers.append(Linear(hidden, hidden))
            layers.append(GELU())
        layers.append(LayerNorm(hidden))
        layers.append(Linear(hidden, n_out))
        self.mlp = Sequential(*layers)
        
    def forward(self, x): 
        return self.mlp(x)


class DeepDecoupledAttention(Module):
    """
    Deep Decoupled Attention (DDA) Model.
    
    This architecture predicts full Gaussian distributions (mu, sigma) 
    for physical properties of galaxies based on dark-matter merger trees.
    Unlike DCA, DDA structurally decouples local and global aggregation into 
    sequential, alternating blocks.
    
    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Latent dimension size.
        out_channels (int): Number of target properties to predict.
        num_pairs (int): Number of alternating Local-Global block pairs.
                         (e.g., num_pairs=6 creates 12 blocks total).
        alpha (float): Residual weight for the local branch.
        lmbda (float): Residual weight for the global branch.
        dropout (float): Dropout probability.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_pairs=6, alpha=0.85, lmbda=0.85, dropout=0.1):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)
        
        # Construct interleaved local/global blocks
        layers = []
        for _ in range(num_pairs):
            layers.append(DDALocalBlock(hidden_channels, alpha=alpha, dropout=dropout))
            layers.append(DDAGlobalBlock(hidden_channels, lmbda=lmbda, num_heads=4, dropout=dropout))
        
        self.blocks = ModuleList(layers)
        
        # Concatenated Global Pooling (Sum, Mean, Max)
        pool_dim = hidden_channels * 3 

        # Probabilistic Decoding Heads
        self.mu_head = MLP(pool_dim, out_channels, hidden=hidden_channels*2)
        self.sig_head = MLP(pool_dim, out_channels, hidden=hidden_channels*2)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        
        x = self.input_proj(x)
        
        # Sequential message passing through decoupled blocks
        for block in self.blocks:
            if isinstance(block, DDALocalBlock):
                x = block(x, edge_index)
            else:
                x = block(x)
                
        # Graph-level embedding
        x_pool = torch.cat([
            global_add_pool(x, batch), 
            global_mean_pool(x, batch), 
            global_max_pool(x, batch)
        ], dim=1)
        
        # Predict parameters of the Gaussian distribution
        mu = self.mu_head(x_pool)
        # Softplus ensures sigma is always strictly positive
        sig = F.softplus(self.sig_head(x_pool)) + 1e-6 
        
        return mu, sig





#Example Usage & Model Loading Guide:
if __name__ == "__main__":
    # Initialize the DDA Model
    model = DeepDecoupledAttention(
       in_channels=16, 
       hidden_channels=128, 
       out_channels=7, 
       num_pairs=6, 
       alpha=0.85, 
       lmbda=0.85
    )
   
    # Load pre-trained weights
    WEIGHTS_PATH = "models/withoutZeros/DDA/model_best_dda.pt"
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint, strict=False)
        print("Weights loaded successfully!")
    

    model.eval()

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    # Load and Evaluate Sample Dataset
    DATASET_PATH = "sample_data/withoutZeros/sample_merger_trees_withoutzeros.pkl"
    
    if os.path.exists(DATASET_PATH):
        sample_graphs = []
        with open(DATASET_PATH, "rb") as f:
            while True:
                try:
                    sample_graphs.append(pickle.load(f))
                except EOFError:
                    break
        
        loader = DataLoader(sample_graphs, batch_size=10, shuffle=False)
        batch = next(iter(loader))
        
        # Pred
        with torch.no_grad():
            mu, sig = model(batch)
        
        print("\n--- DDA Prediction Results (first batch) ---")
        print(f"Graph num: {batch.num_graphs}")
        print(f"Mu Matrix (Predictions) : {mu.shape}")
        print(f"Sigma Matrix (Uncertainty) : {sig.shape}")
        
        print("\Output example (All targets):")
        with np.printoptions(precision=4, suppress=True):
            print(f"Mu    : {mu[0].numpy()}")
            print(f"Sigma : {sig[0].numpy()}")