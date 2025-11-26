import math
import torch
import torch.nn as nn


class EpochTransformer(nn.Module):
    """
    Transformer-based model for epoch-level classification.
    Uses a learnable CLS token for classification.
    Patch reduction with mean-pooling.
    """
    def __init__(
        self,
        input_channels=2,      # N_HB_CHANNELS from your dataset
        seq_length=7680,       # Expected sequence length (e.g., 7680 for 256Hz*30s)
        d_model=64,            # embedding dimension
        nhead=8,               # number of attention heads
        num_layers=4,          # number of transformer layers
        dim_feedforward=256,   # MLP hidden dimension
        dropout=0.1,
        num_classes=5,         # Wake, N1, N2, N3, REM
    ):
        """
        Parameters
        ----------
        seq_length : int
            Original sequence length per epoch (e.g., 7680 for 256Hz, 3840 for 128Hz).
            Will be reduced via patching to ≤1024 tokens.
        """
        super().__init__()
        
        self.d_model = d_model
        self.original_seq_length = seq_length

        # Compute patch size to get ≤1024 tokens
        MAX_TOKENS = 512   # SHOULD BE in {1024, 512, 256}!
        self.patch_size = math.ceil(seq_length / MAX_TOKENS)
        self.final_seq_length = seq_length // self.patch_size

        # Assert exact division (no truncation needed)
        assert seq_length % self.patch_size == 0, (
            f"Sequence length {seq_length} not divisible by patch size {self.patch_size}. "
            f"This will cause information loss. Adjust preprocessing to ensure divisibility."
        )

        # Print info for transparency
        print(f"\nPatch reduction in model:")
        print(f"  Original sequence length: {self.original_seq_length}")
        print(f"  Patch size (mean pool): {self.patch_size}")
        print(f"  Final sequence length (tokens): {self.final_seq_length}")
        print(f"  Reduction factor: {self.patch_size}x\n")
        
        # Project input channels to d_model
        self.input_projection = nn.Linear(input_channels, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding for final_seq_length + 1 (including CLS)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.final_seq_length + 1, d_model))
        
        # PyTorch's TransformerEncoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: expects (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def _apply_patch_mean_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply mean-pooling patches to reduce sequence length.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, channels, time)
            Input with original sequence length.
        
        Returns
        -------
        torch.Tensor, shape (batch, channels, final_seq_length)
            Patched input with reduced sequence length.
        """
        batch_size, channels, time = x.shape
        
        # Validate input length
        if time != self.original_seq_length:
            raise ValueError(
                f"Input time dimension {time} doesn't match expected "
                f"original_seq_length {self.original_seq_length}"
            )
        
        ### no trim, since we'd lose information. We assert exact divisibility in __init__ ###

        # Reshape: (batch, channels, final_seq_length, patch_size)
        x = x.reshape(batch_size, channels, self.final_seq_length, self.patch_size)
        
        # Mean over patch dimension: (batch, channels, final_seq_length)
        x = x.mean(dim=-1)
        
        return x

    def forward(self, x):
        """
        Forward pass with patch mean-pooling.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, channels, time)
            Input EEG data with original sequence length.
        
        Returns
        -------
        torch.Tensor, shape (batch, num_classes)
            Class logits.
        """
        # x: (batch, channels, original_seq_length)
        batch_size = x.size(0)

        # Apply patch mean-pooling: (batch, channels, original_seq_length) 
        #                         -> (batch, channels, final_seq_length)
        x = self._apply_patch_mean_pooling(x)
        
        # Transpose for projection: (batch, final_seq_length, channels)
        x = x.transpose(1, 2)
        
        # Project to d_model: (batch, final_seq_length, d_model)
        x = self.input_projection(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, final_seq_length+1, d_model)
        
        # Add positional encoding
        x = x + self.pos_embedding
        
         # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, final_seq_length+1, d_model)
        
        # Extract CLS token output (first position)
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        # Classify
        logits = self.classifier(cls_output)  # (batch, num_classes)
        return logits