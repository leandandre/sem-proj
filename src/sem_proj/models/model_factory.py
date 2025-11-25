import torch
import torch.nn as nn


class EpochTransformer(nn.Module):
    """
    Transformer-based model for epoch-level classification.
    Uses a learnable CLS token for classification.
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
            Expected sequence length per epoch. This should match your preprocessing:
            - 7680 for 256 Hz * 30s (no resampling)
            - 3840 for 128 Hz * 30s (resampled to 128 Hz)
        """
        super().__init__()
        
        self.d_model = d_model
        self.seq_length = seq_length
        
        # Project input channels to d_model
        self.input_projection = nn.Linear(input_channels, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding for seq_length + 1 (including CLS token)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length + 1, d_model))
        
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

    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, channels, time)
            Input EEG data. Time dimension must match self.seq_length.
        
        Returns
        -------
        torch.Tensor, shape (batch, num_classes)
            Class logits.
        """
        # x: (batch, channels, time)
        batch_size = x.size(0)
        seq_length = x.size(2)
        
        # Validate sequence length matches expected
        if seq_length != self.seq_length:
            raise ValueError(
                f"Input sequence length {seq_length} does not match model's expected "
                f"seq_length {self.seq_length}. Check your preprocessing configuration."
            )
        
        x = x.transpose(1, 2)  # (batch, time, channels)
        
        # Project to d_model
        x = self.input_projection(x)  # (batch, time, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, time+1, d_model)
        
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Apply transformer
        x = self.transformer_encoder(x)  # (batch, time+1, d_model)
        
        # Extract CLS token output (first position)
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        # Classify
        logits = self.classifier(cls_output)  # (batch, num_classes)
        return logits