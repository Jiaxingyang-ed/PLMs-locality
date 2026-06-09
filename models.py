"""
Shared model classes for locality probing experiments.

This module contains the core model architectures used across the project:
- LocalSeqEncoder: Basic local sequence encoder for ProtT5 embeddings
- LocalSeqEncoderESM: Local sequence encoder for ESM-2 embeddings
- DeepLocalSeqEncoder: Enhanced version with deeper architecture and ESM-2 initialization
- DeltaDataset: Dataset class for training local encoders
"""

import torch
from torch import nn
from torch.utils.data import Dataset


class LocalSeqEncoder(nn.Module):
    """
    Basic local sequence encoder to predict global residue embeddings from local windows.
    
    This model takes a local sequence window and predicts the global embedding delta
    for the central residue. Used for ProtT5 embeddings (1024-dimensional).
    
    Args:
        vocab_size: Number of amino acid tokens (default: 20)
        embed_dim: Dimension of token embeddings (default: 64)
        hidden_dim: Hidden dimension in transformer (default: 128)
        output_dim: Output embedding dimension (default: 1024 for ProtT5)
    """
    
    def __init__(self, vocab_size=20, embed_dim=64, hidden_dim=128, output_dim=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, seq_idx, mut_idx=None):
        """
        Forward pass.
        
        Args:
            seq_idx: Token indices of shape (batch, seq_len)
            mut_idx: Mutation token index (optional, not used in basic version)
        
        Returns:
            Predicted embedding of shape (batch, output_dim)
        """
        x = self.embed(seq_idx)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


class LocalSeqEncoderESM(nn.Module):
    """
    Local sequence encoder for ESM-2 embeddings.
    
    Similar to LocalSeqEncoder but configured for ESM-2's 640-dimensional embeddings.
    
    Args:
        vocab_size: Number of amino acid tokens (default: 20)
        embed_dim: Dimension of token embeddings (default: 64)
        hidden_dim: Hidden dimension in transformer (default: 128)
        output_dim: Output embedding dimension (default: 640 for ESM-2)
    """
    
    def __init__(self, vocab_size=20, embed_dim=64, hidden_dim=128, output_dim=640):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=4,
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, seq_idx, mut_idx=None):
        """
        Forward pass.
        
        Args:
            seq_idx: Token indices of shape (batch, seq_len)
            mut_idx: Mutation token index (optional)
        
        Returns:
            Predicted embedding of shape (batch, output_dim)
        """
        x = self.embed(seq_idx)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


class DeepLocalSeqEncoder(nn.Module):
    """
    Enhanced local sequence encoder with deeper architecture.
    
    Features:
    - Deeper transformer (4 layers by default)
    - Positional encoding
    - Optional ESM-2 weight initialization
    - Support for variable window sizes
    
    Args:
        vocab_size: Number of amino acid tokens (default: 20)
        embed_dim: Dimension of token embeddings (default: 64)
        hidden_dim: Hidden dimension in transformer (default: 256)
        output_dim: Output embedding dimension (default: 1024)
        num_layers: Number of transformer encoder layers (default: 4)
        nhead: Number of attention heads (default: 4)
        max_window_len: Maximum sequence window length (default: 41)
        use_esm_init: Whether to initialize with ESM-2 embeddings (default: False)
    """
    
    def __init__(self, vocab_size=20, embed_dim=64, hidden_dim=256, output_dim=1024,
                 num_layers=4, nhead=4, max_window_len=41, use_esm_init=False):
        super().__init__()
        self.max_window_len = max_window_len
        
        # Token embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Optional ESM-2 weight initialization
        if use_esm_init:
            try:
                from transformers import EsmModel
                esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
                # Copy embedding weights (note: ESM-2 vocab includes special tokens)
                self.embed.weight.data[:vocab_size] = esm.embeddings.word_embeddings.weight.data[:vocab_size]
                print("ESM-2 embedding weights loaded successfully")
            except Exception as e:
                print(f"ESM-2 loading failed, using random initialization: {e}")
        
        # Positional encoding
        self.pos_embed = nn.Embedding(max_window_len, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead,
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, seq_idx, mask=None):
        """
        Forward pass.
        
        Args:
            seq_idx: Token indices of shape (batch, seq_len)
            mask: Optional padding mask for transformer
        
        Returns:
            Predicted embedding of shape (batch, output_dim)
        """
        batch, seq_len = seq_idx.shape
        x = self.embed(seq_idx)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=seq_idx.device).unsqueeze(0).expand(batch, -1)
        x = x + self.pos_embed(positions)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Mean pooling
        x = x.mean(dim=1)  # (batch, embed_dim)
        
        return self.fc(x)  # (batch, output_dim)


class DeltaDataset(Dataset):
    """
    Dataset for training local sequence encoders.
    
    This dataset handles mutation data with local sequence windows and global embedding deltas.
    Supports both ProtT5 and ESM-2 formats.
    
    Args:
        data: List of dictionaries containing mutation data
        max_len: Maximum sequence window length (default: 21)
        include_mut_idx: Whether to include mutation amino acid index (default: True)
    """
    
    def __init__(self, data, max_len=21, include_mut_idx=True):
        self.max_len = max_len
        self.include_mut_idx = include_mut_idx
        self.samples = []
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        
        for d in data:
            local_seq = d['local_seq']
            delta_h = d['delta_h_global']
            seq_idx = [self.aa_to_idx.get(aa, 0) for aa in local_seq]
            
            # Pad or truncate to max_len
            if len(seq_idx) < max_len:
                seq_idx += [0] * (max_len - len(seq_idx))
            else:
                seq_idx = seq_idx[:max_len]
            
            if include_mut_idx and 'mut_aa' in d:
                mut_idx = self.aa_to_idx.get(d['mut_aa'], 0)
                self.samples.append((torch.tensor(seq_idx), mut_idx, torch.tensor(delta_h)))
            else:
                self.samples.append((torch.tensor(seq_idx, dtype=torch.long),
                                   torch.tensor(delta_h, dtype=torch.float)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
