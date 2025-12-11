"""
Variable-length sequence models for sleep stage classification.

Extends existing models to handle variable-length sequences with padding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceGRUClassifier_Variable(nn.Module):
    """
    GRU-based sequence classifier with support for variable-length sequences.
    
    Uses pack_padded_sequence for efficient processing of padded batches.
    Designed for use with BoasVariableLengthSequenceDataset.
    
    Architecture:
    1. Encode each epoch with pretrained epoch encoder → embeddings
    2. Pack variable-length sequences
    3. Process with bidirectional GRU
    4. Classify each epoch in sequence (many-to-many)
    """
    
    def __init__(
        self,
        epoch_model,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        """
        Parameters
        ----------
        epoch_model : nn.Module
            Pretrained epoch encoder (e.g., EpochTransformerConv1D_v2).
            Should have return_mean_embedding=True capability.
        hidden_size : int
            GRU hidden size.
        num_layers : int
            Number of GRU layers.
        num_classes : int
            Number of sleep stage classes (5: Wake, N1, N2, N3, REM).
        bidirectional : bool
            Whether to use bidirectional GRU.
        dropout : float
            Dropout between GRU layers (only used if num_layers > 1).
        """
        super().__init__()
        
        self.epoch_model = epoch_model
        self.input_dim = epoch_model.d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        
        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Classifier head
        gru_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Linear(gru_output_dim, num_classes)
        
        # Dropout before classifier
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, lengths=None):
        """
        Forward pass with variable-length sequences.
        
        Parameters
        ----------
        x : torch.Tensor
            Padded sequences (B, L_max, C, T).
        lengths : torch.Tensor
            Actual sequence lengths (B,). If None, assumes no padding.
        
        Returns
        -------
        logits : torch.Tensor
            Classification logits (B, L_max, num_classes).
            Padded positions will have arbitrary values (should be masked during loss).
        """
        B, L_max, C, T = x.shape
        
        # 1) Encode all epochs (including padding - will be masked later)
        x_flat = x.view(B * L_max, C, T)  # (B*L_max, C, T)
        
        with torch.set_grad_enabled(self.training):
            # Get epoch embeddings
            emb_flat = self.epoch_model(x_flat, return_mean_embedding=True)  # (B*L_max, d_model)
        
        # 2) Reshape to sequence form
        emb_seq = emb_flat.view(B, L_max, -1)  # (B, L_max, d_model)
        
        # 3) Pack sequences for efficient GRU processing
        if lengths is not None:
            # Sort sequences by length (descending) - required for pack_padded_sequence
            # Note: collate_fn already sorts, but we ensure it here
            lengths_cpu = lengths.cpu()
            
            # Pack the embeddings
            packed_emb = nn.utils.rnn.pack_padded_sequence(
                emb_seq, lengths_cpu, batch_first=True, enforce_sorted=True
            )
            
            # Process with GRU
            packed_out, _ = self.gru(packed_emb)
            
            # Unpack back to padded format
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=L_max
            )
        else:
            # No padding - process normally
            gru_out, _ = self.gru(emb_seq)  # (B, L_max, hidden_size * directions)
        
        # 4) Apply dropout and classify
        gru_out = self.dropout(gru_out)
        logits = self.classifier(gru_out)  # (B, L_max, num_classes)
        
        return logits
    
    def freeze_epoch_encoder(self):
        """Freeze the epoch encoder parameters."""
        for param in self.epoch_model.parameters():
            param.requires_grad = False
        print("Epoch encoder frozen.")
    
    def unfreeze_epoch_encoder(self):
        """Unfreeze the epoch encoder parameters."""
        for param in self.epoch_model.parameters():
            param.requires_grad = True
        print("Epoch encoder unfrozen.")


class SequenceGRUClassifier_VariableWithAttention(nn.Module):
    """
    Enhanced GRU classifier with self-attention mechanism.
    
    Adds attention layer after GRU to better capture long-range dependencies
    in variable-length sleep sequences.
    """
    
    def __init__(
        self,
        epoch_model,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        bidirectional: bool = True,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        """
        Parameters
        ----------
        epoch_model : nn.Module
            Pretrained epoch encoder.
        hidden_size : int
            GRU hidden size.
        num_layers : int
            Number of GRU layers.
        num_classes : int
            Number of classes.
        bidirectional : bool
            Bidirectional GRU.
        dropout : float
            Dropout rate.
        use_attention : bool
            Whether to use self-attention after GRU.
        """
        super().__init__()
        
        self.epoch_model = epoch_model
        self.input_dim = epoch_model.d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # GRU
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        gru_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Self-attention (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=gru_output_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True,
            )
            self.layer_norm = nn.LayerNorm(gru_output_dim)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_output_dim, num_classes)
    
    def forward(self, x, lengths=None):
        """
        Forward pass with optional attention.
        
        Parameters
        ----------
        x : torch.Tensor
            Padded sequences (B, L_max, C, T).
        lengths : torch.Tensor
            Actual sequence lengths (B,).
        
        Returns
        -------
        logits : torch.Tensor
            (B, L_max, num_classes)
        """
        B, L_max, C, T = x.shape
        
        # 1) Encode epochs
        x_flat = x.view(B * L_max, C, T)
        emb_flat = self.epoch_model(x_flat, return_mean_embedding=True)
        emb_seq = emb_flat.view(B, L_max, -1)
        
        # 2) GRU processing
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed_emb = nn.utils.rnn.pack_padded_sequence(
                emb_seq, lengths_cpu, batch_first=True, enforce_sorted=True
            )
            packed_out, _ = self.gru(packed_emb)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=L_max
            )
        else:
            gru_out, _ = self.gru(emb_seq)
        
        # 3) Optional self-attention
        if self.use_attention:
            # Create attention mask for padding
            if lengths is not None:
                # Mask: True for positions to IGNORE
                attn_mask = torch.arange(L_max, device=x.device)[None, :] >= lengths[:, None]
            else:
                attn_mask = None
            
            # Self-attention with residual connection
            attn_out, _ = self.attention(
                gru_out, gru_out, gru_out,
                key_padding_mask=attn_mask,
            )
            gru_out = self.layer_norm(gru_out + attn_out)
        
        # 4) Classify
        gru_out = self.dropout(gru_out)
        logits = self.classifier(gru_out)
        
        return logits
    
    def freeze_epoch_encoder(self):
        """Freeze the epoch encoder."""
        for param in self.epoch_model.parameters():
            param.requires_grad = False
    
    def unfreeze_epoch_encoder(self):
        """Unfreeze the epoch encoder."""
        for param in self.epoch_model.parameters():
            param.requires_grad = True


# Export for easy importing
__all__ = [
    'SequenceGRUClassifier_Variable',
    'SequenceGRUClassifier_VariableWithAttention',
]