# Pay More Attention to your Neighbors: Local vs. Global Graph Attention for Learning Galaxy-Properties from Merger Trees

This repository contains the official PyTorch implementation of the models proposed in the paper: **"Pay More Attention to your Neighbors: Local vs. Global Graph Attention for Learning Galaxy-Properties from Merger Trees"**.

## Architectures

1. **Deep Coupled Attention (DCA):** Stacks local and global attention mechanisms, allowing the network to either learn the optimal balance dynamically via a Softmax gating mechanism (`learnable_alpha=True`) or use a fixed hyperparameter.
2. **Deep Decoupled Attention (DDA):** A sequential architecture that structurally decouples local (GraphSAGE) and global (Latent Attention) aggregation, alternating blocks.

## Repository Structure

* `dca_model.py`: Contains the `DeepCoupledAttention` class.
* `dda_model.py`: Contains the `DeepDecoupledAttention` class.
* `models/`: Contains the pre-trained `.pt` files for the best performing models (DCA Learnable, DCA Hypertuned, and DDA) for both dataset types (with zeros and without zeros).
* `sample_data/`: A small subset of formatted merger trees to test the forward pass.
