import torch
import os
import pickle
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Module, Sequential, ModuleList, Dropout, GELU
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_add_pool, global_max_pool, global_mean_pool



# Deep Coupled Attention (DCA) Architecture

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


class DCABlock(Module):
    """
    A single block of the Deep Coupled Attention (DCA) architecture.
    Couples Global (Latent Attention) and Local (GraphSAGE) information, 
    followed by a non-linear FFN to enable deep architectural stacking.
    """
    def __init__(self, hidden_channels, lmbda=0.85, num_heads=4, dropout=0.1):
        super().__init__()
        self.lmbda = lmbda
        self.norm = LayerNorm(hidden_channels)
        
        # Parallel attention branches
        self.attn_net = LatentGlobalAttention(hidden_channels, hidden_channels, num_heads)
        self.local_gnn = SAGEConv(hidden_channels, hidden_channels)
        
        # FFN for non-linearity
        self.ffn = Sequential(
            Linear(hidden_channels, hidden_channels * 2),
            GELU(),
            Dropout(dropout),
            Linear(hidden_channels * 2, hidden_channels)
        )
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, w_global, w_local):
        h = self.norm(x)
        
        # Global Branch with residual lmbda
        z_global_raw = self.attn_net(h)
        z_global = self.lmbda * z_global_raw + (1 - self.lmbda) * h 
        
        # Local Branch (1-hop neighborhood)
        z_local = F.relu(self.local_gnn(h, edge_index))
        
        # Coupling mechanism governed by alpha weights
        z_out = w_global * z_global + w_local * z_local
        
        # Residual connection over the Feed-Forward Network
        x = x + self.dropout(self.ffn(z_out))
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


class DeepCoupledAttention(Module):
    """
    Deep Coupled Attention (DCA) Model.
    
    This architecture predicts full Gaussian distributions (mu, sigma) 
    for physical properties of galaxies based on dark-matter merger trees.
    
    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Latent dimension size.
        out_channels (int): Number of target properties to predict.
        num_blocks (int): Number of stacked DCA blocks (depth).
        lmbda (float): Residual weight for the global attention branch.
        alpha (float): Initial or fixed weight for local attention (w_local).
                       w_global will be (1 - alpha).
        learnable_alpha (bool): If True, alpha becomes a learnable parameter
                                dynamically optimized during training via Softmax.
        dropout (float): Dropout probability.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_blocks=6, lmbda=0.85, alpha=0.5, learnable_alpha=False, dropout=0.1):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)
        self.learnable_alpha = learnable_alpha
        
        # Initialization of alpha parameter mechanism
        if self.learnable_alpha:
            # Starts at [1.0, 1.0] -> Softmax makes it [0.5, 0.5]
            self.alpha_weights = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        else:
            self.alpha_fixed = alpha
            
        self.blocks = ModuleList([
            DCABlock(hidden_channels, lmbda=lmbda, num_heads=4, dropout=dropout) 
            for _ in range(num_blocks)
        ])
        
        # Concatenated Global Pooling (Sum, Mean, Max)
        pool_dim = hidden_channels * 3 
        
        # Probabilistic Decoding Heads
        self.mu_head = MLP(pool_dim, out_channels, hidden=hidden_channels * 2)
        self.sig_head = MLP(pool_dim, out_channels, hidden=hidden_channels * 2)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.input_proj(x)
        
        # Determine current attention weights
        if self.learnable_alpha:
            weights = F.softmax(self.alpha_weights, dim=0)
            w_global, w_local = weights[0], weights[1]
        else:
            w_global = 1.0 - self.alpha_fixed
            w_local = self.alpha_fixed
            
        # Expanding k-hop
        for block in self.blocks:
            x = block(x, edge_index, w_global, w_local)
        
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
    # Initialize the DCA Model
    model = DeepCoupledAttention(
        in_channels=16,
        hidden_channels=128, 
        out_channels=7,
        learnable_alpha=True
    )
    
    # Load pre-trained weights
    WEIGHTS_PATH = "models/withZeros/DCA/Learnable/model_best_dca_learnable.pt" 
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint, strict=False)
        print("Weights loaded successfully!")


    model.eval()

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    # Load and Evaluate Sample Dataset
    DATASET_PATH = "sample_data/withZeros/sample_merger_trees_withzeros.pkl"
    
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
        
        print("\n--- DCA Prediction Results (first batch) ---")
        print(f"Graph num: {batch.num_graphs}")
        print(f"Mu Matrix (Predictions) : {mu.shape}")
        print(f"Sigma Matrix (Uncertainty) : {sig.shape}")
        
        print("\Output example (All targets):")
        with np.printoptions(precision=4, suppress=True):
            print(f"Mu    : {mu[0].numpy()}")
            print(f"Sigma : {sig[0].numpy()}")