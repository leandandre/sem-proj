import math
from matplotlib.pylab import rint
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class EpochTransformerConv1D_v2(nn.Module):
    def __init__(
        self,
        input_channels=2,
        seq_length=7680,
        d_model=64,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        num_classes=5,
        target_tokens: int = 480
    ):
        super().__init__()
        self.d_model = d_model
        self.original_seq_length = seq_length
        self.target_tokens = target_tokens
        assert target_tokens in {480, 240}, "target_tokens must be 480 or 240"
        if target_tokens == 480:
            self.tokenization = nn.Sequential(
                nn.Conv1d(in_channels=input_channels,
                        out_channels=32,
                        kernel_size=5,
                        stride=2,
                        padding=2),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(in_channels=32,
                        out_channels=64,
                        kernel_size=5,
                        stride=2,
                        padding=2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(in_channels=64,
                        out_channels=128,
                        kernel_size=5,
                        stride=2,
                        padding=2),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(in_channels=128,
                        out_channels=d_model,
                        kernel_size=1,
                        stride=1,
                        padding=0),
            )
        else:  # target_tokens == 240
            self.tokenization = nn.Sequential(
                nn.Conv1d(in_channels=input_channels,
                        out_channels=32,
                        kernel_size=5,
                        stride=4,       # only change (faster reduction)
                        padding=2),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(in_channels=32,
                        out_channels=64,
                        kernel_size=5,
                        stride=2,
                        padding=2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(in_channels=64,
                        out_channels=128,
                        kernel_size=5,
                        stride=2,
                        padding=2),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(in_channels=128,
                        out_channels=d_model,
                        kernel_size=1,
                        stride=1,
                        padding=0),
            )
        self.pos_embedding = nn.Parameter(torch.randn(1, target_tokens, d_model))   # no CLS for now
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

    
    def forward(self, x, return_mean_embedding=False):
        x = self.tokenization(x)  # (batch, d_model, target_tokens)
        x = x.transpose(1, 2)   # (batch, target_tokens, d_model)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)  # (batch, target_tokens, d_model)
        mean = x.mean(dim=1)  # mean pooling over tokens, no CLS for now
        if return_mean_embedding:
            return mean  # (batch, d_model), model can now be used for inter-epoch sequence modeling
        logits = self.classifier(mean)  # (batch, num_classes)
        return logits
    

### feeding several epoch embeddings into a sequence model (here, GRU) ###
class SequenceGRUClassifier(nn.Module):
    def __init__(self, epoch_model, hidden_size=128, num_layers=1, num_classes=5, bidirectional=False):
        super().__init__()
        self.epoch_model = epoch_model          # EpochTransformerConv1d_v2
        self.input_dim = epoch_model.d_model    # d_model (e.g. 64)
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,      # input (B, L, D)
            dropout=0.2,
            bidirectional=bidirectional     # try bidir --> change classifier input size (*2)
        )
        gru_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Linear(gru_output_dim, num_classes)

    def forward(self, x):
        """
        x: (B, L, C, T)
        returns: logits (B, L, num_classes)
        """
        B, L, C, T = x.shape

        # 1) Flatten epochs within batch
        x_flat = x.view(B * L, C, T)                  # (B*L, C, T)

        # 2) Encode each epoch -> embedding
        emb_flat = self.epoch_model(x_flat, return_mean_embedding=True)  # (B*L, d_model)

        # 3) Reshape back to sequence form
        emb_seq = emb_flat.view(B, L, -1)             # (B, L, d_model)

        # 4) GRU over epochs
        gru_out, _ = self.gru(emb_seq)                # (B, L, hidden_size)

        # 5) Classification for each epoch (many-to-many)
        logits = self.classifier(gru_out)             # (B, L, num_classes)
        return logits


class SequenceTransformerClassifier(nn.Module):
    """
    Sequence-level classifier using a small Transformer instead of GRU.
    
    Architecture:
    1. Encode each epoch with EpochTransformerConv1D_v2 → (B*L, d_model)
    2. Reshape to sequence form → (B, L, d_model)
    3. Pass through Transformer encoder layers → (B, L, d_model)
    4. Classify each epoch in the sequence → (B, L, num_classes)
    
    Advantages over GRU:
    - Parallel computation (faster)
    - Better long-range dependencies
    - Attention weights interpretable
    """
    def __init__(
        self,
        epoch_model,
        d_model_seq: int = 96,      # Embedding dimension for sequence transformer
        nhead: int = 4,              # Number of attention heads
        num_layers: int = 2,         # Number of transformer layers
        dim_feedforward: int = 384,  # Feedforward dimension
        dropout: float = 0.2,
        num_classes: int = 5,
    ):
        """
        Parameters
        ----------
        epoch_model : EpochTransformerConv1D_v2
            Pre-trained or trainable epoch encoder.
        d_model_seq : int
            Embedding dimension for the sequence transformer.
            Will project epoch embeddings to this dimension.
        nhead : int
            Number of attention heads in sequence transformer.
        num_layers : int
            Number of transformer encoder layers for sequence modeling.
        dim_feedforward : int
            Feedforward dimension in transformer layers.
        dropout : float
            Dropout rate.
        num_classes : int
            Number of output classes.
        """
        super().__init__()
        self.epoch_model = epoch_model
        self.input_dim = epoch_model.d_model  # d_model from epoch model (e.g., 64)
        self.d_model_seq = d_model_seq
        
        # Project epoch embeddings to sequence transformer dimension
        self.embedding_projection = nn.Linear(self.input_dim, d_model_seq)
        
        # Positional encoding for sequence transformer
        # Max sequence length (assumption: won't exceed 100 epochs in a sequence)
        max_seq_len = 100
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model_seq))
        
        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model_seq,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head for each epoch
        self.classifier = nn.Linear(d_model_seq, num_classes)

    def forward(self, x):
        """
        Forward pass for sequence classification.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequences with shape (B, L, C, T) where:
            - B: batch size
            - L: sequence length (number of epochs)
            - C: number of channels
            - T: number of timepoints per epoch
        
        Returns
        -------
        torch.Tensor
            Classification logits with shape (B, L, num_classes)
        """
        B, L, C, T = x.shape

        # 1) Flatten epochs within batch
        x_flat = x.view(B * L, C, T)  # (B*L, C, T)

        # 2) Encode each epoch -> embedding
        emb_flat = self.epoch_model(x_flat, return_mean_embedding=True)  # (B*L, d_model)

        # 3) Project to sequence transformer dimension
        emb_proj = self.embedding_projection(emb_flat)  # (B*L, d_model_seq)

        # 4) Reshape back to sequence form
        emb_seq = emb_proj.view(B, L, -1)  # (B, L, d_model_seq)

        # 5) Add positional encoding
        emb_seq = emb_seq + self.pos_embedding[:, :L, :]  # (B, L, d_model_seq)

        # 6) Transformer encoder over epochs
        trans_out = self.transformer_encoder(emb_seq)  # (B, L, d_model_seq)

        # 7) Classification for each epoch (many-to-many)
        logits = self.classifier(trans_out)  # (B, L, num_classes)
        return logits



    




class SSLEpochTransformerConv1D(nn.Module):
    """
    Basically a copy from SL version, but without CLS token and classification head.
    Outputs the full transformer output for SSL purposes.
    """
    def __init__(
        self,
        input_channels=2,      # N_HB_CHANNELS or N_PSG_CHANNELS from your dataset
        seq_length=7680,       # Expected sequence length (e.g., 7680 for 256Hz*30s)
        d_model=64,            # embedding dimension
        nhead=8,               # number of attention heads
        num_layers=4,          # number of transformer layers
        dim_feedforward=256,   # MLP hidden dimension
        dropout=0.1,
        num_classes=5,         # will be ignored in SSL 
        max_tokens: int = 512  # SHOULD BE in {1024, 512, 256}!
    ):
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
        # Positional encoding for final_seq_length (no CLS token)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.final_seq_length, d_model))
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

    def forward(self, x):
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
       
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, final_seq_length, d_model)

        # return whole transformer output for SSL purposes
        return x

class SSLClassifierHead(nn.Module):
    """
    Simple classification head for SSL transformer outputs for fine-tuning.
    """
    def __init__(
        self,
        d_model=64,            # embedding dimension from the transformer
        dropout=0.1,
        num_classes=5,         # Wake, N1, N2, N3, REM
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        Forward pass for classification head.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, d_model)
            Mean of transformer output.
        
        Returns
        -------
        torch.Tensor, shape (batch, num_classes)
            Class logits based on the mean of transformer outputs.
        """
        logits = self.classifier(x)  # (batch, num_classes)
        return logits