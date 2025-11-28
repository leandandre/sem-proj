import math
from matplotlib.pylab import rint
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
        input_channels=2,      # N_HB_CHANNELS from the dataset
        seq_length=7680,       # Expected sequence length (e.g., 7680 for 256Hz*30s)
        d_model=64,            # embedding dimension
        nhead=8,               # number of attention heads
        num_layers=4,          # number of transformer layers
        dim_feedforward=256,   # MLP hidden dimension
        dropout=0.1,
        num_classes=5,         # Wake, N1, N2, N3, REM
        max_tokens: int = 512  # SHOULD BE in {1024, 512, 256}!
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
        self.max_tokens = max_tokens

        # Compute patch size to get less than max_tokens tokens
        self.patch_size = math.ceil(seq_length / self.max_tokens)
        self.final_seq_length = seq_length // self.patch_size

        # Assert exact division (no truncation needed)
        assert seq_length % self.patch_size == 0, (
            f"Sequence length {seq_length} not divisible by patch size {self.patch_size}. "
            f"This will cause information loss. Adjust preprocessing to ensure divisibility."
        )

        # Print info for transparency
        print(f"\nPatch reduction in model (MeanPooling):")
        print(f"  Original sequence length: {self.original_seq_length}")
        print(f"  max_tokens: {self.max_tokens}")
        print(f"  Patch size (mean pool): {self.patch_size}")
        print(f"  Final sequence length (tokens): {self.final_seq_length}\n")
        
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
            batch_first=True  # expects (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
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
        # print(f"before pooling: {x.shape}")
        # Apply patch mean-pooling: (batch, channels, original_seq_length) 
        #                         -> (batch, channels, final_seq_length)
        x = self._apply_patch_mean_pooling(x)
        # print(f"after pooling: {x.shape}")
        
        # Transpose for projection: (batch, final_seq_length, channels)
        x = x.transpose(1, 2)
        
        # Project to d_model: (batch, final_seq_length, d_model)
        x = self.input_projection(x)
        # print(f"after input projection: {x.shape}")

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
    


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))   


class EpochTransformerConv1D(nn.Module):
    """
    Transformer-based model for epoch-level classification.
    Uses 1D convolution for patch embedding instead of mean-pooling.
    Each patch is directly projected to d_model via Conv1D.
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
        max_tokens: int = 512  # SHOULD BE in {1024, 512, 256}!
    ):
        """
        Parameters
        ----------
        seq_length : int
            Original sequence length per epoch (e.g., 7680 for 256Hz, 3840 for 128Hz).
            Will be reduced via Conv1D to ≤1024 tokens.
        """
        super().__init__()
        
        self.d_model = d_model
        self.original_seq_length = seq_length
        self.max_tokens = max_tokens

        # Compute patch size to get ≤ max_tokens tokens
        self.patch_size = math.ceil(seq_length / self.max_tokens)
        self.final_seq_length = seq_length // self.patch_size

        # Assert exact division (no truncation needed)
        assert seq_length % self.patch_size == 0, (
            f"Sequence length {seq_length} not divisible by patch size {self.patch_size}. "
            f"This will cause information loss. Adjust preprocessing to ensure divisibility."
        )

        # Print info for transparency
        print(f"\nPatch reduction in model (Conv1D):")
        print(f"  Original sequence length: {self.original_seq_length}")
        print(f"  Patch size (conv kernel): {self.patch_size}")
        print(f"  Final sequence length (tokens): {self.final_seq_length}")
        print(f"  Reduction factor: {self.patch_size}x\n")
        
        # 1D Convolution for patch embedding
        # Input: (batch, input_channels, seq_length)
        # Output: (batch, d_model, final_seq_length)
        self.patch_embedding = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,  # Non-overlapping patches
            padding=0,
            bias=True
        )

        # Optional: additional Conv1D layer(s) to refine token representations
        self.token_refine = nn.Sequential(
            ResidualConvBlock(d_model, kernel_size=3, dilation=1, dropout=dropout),
            ResidualConvBlock(d_model, kernel_size=3, dilation=1, dropout=dropout),
            # ResidualConvBlock(d_model, kernel_size=3, dilation=2, dropout=dropout),
        )

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
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        Forward pass with Conv1D patch embedding.
        
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

        # Validate input length
        if x.size(2) != self.original_seq_length:
            raise ValueError(
                f"Input time dimension {x.size(2)} doesn't match expected "
                f"original_seq_length {self.original_seq_length}"
            )

        # Apply Conv1D patch embedding: (batch, channels, seq_length) 
        #                             -> (batch, d_model, final_seq_length)
        # print(f"before patch embedding: {x.shape}")
        x = self.patch_embedding(x)
        # print(f"after patch embedding: {x.shape}")
        x = self.token_refine(x)
        # print(f"after token refine: {x.shape}")
        
        # Transpose for transformer: (batch, final_seq_length, d_model)
        x = x.transpose(1, 2)

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