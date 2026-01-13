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
    



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 29):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)          # (T, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                        # (1, T, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiChannelSleepNet(nn.Module):
    def __init__(
        self,
        num_channels: int = 2,
        num_classes: int = 5,
        fs: int = 128,               # sampling frequency (Hz)
        epoch_len_sec: int = 30,     # each window is 30s
        n_fft: int = 256,            # FFT size
        num_head: int = 4,
        forward_hidden: int = 512,  # FFN hidden dim inside transformer, maybe increase to 768
        num_encoder: int = 3,       # single-channel transformer depth (number of layers)
        num_encoder_multi: int = 2,  # multichannel transformer depth
        fc_hidden: int = 512,       # classifier hidden dim, maybe increase to 768
        dropout_tf: float = 0.1,     # dropout before multi-block & in FC
        dropout_tr: float = 0.1,     # dropout inside transformers & PE
    ):
        super().__init__()
    
        # --- STFT / time-frequency parameters ---
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.fs = fs
        self.epoch_len_sec = epoch_len_sec
        self.n_fft = n_fft

        # STFT window & hop (2s window, 1s overlap)
        self.win_length = int(2 * fs)   # samples per window
        self.hop_length = int(1 * fs)   # stride between windows
        assert self.win_length <= self.n_fft, (
            "win_length must be <= n_fft (pad/truncate if needed)."
        )

        # Frequency bins after dropping DC:  n_fft/2
        self.freq_bins = self.n_fft // 2  # e.g. 128
        self.dim_model = self.freq_bins   # transformer feature dim

        # Number of time steps (frames) for 30s
        # T = floor((L - win_length) / hop_length) + 1
        L_expected = epoch_len_sec * fs
        self.pad_size = int((L_expected - self.win_length) / self.hop_length) + 1

        # STFT window (Hann)
        self.register_buffer(
            "stft_window",
            torch.hann_window(self.win_length),
            persistent=False,
        )

        # --- Single-channel transformer block ---
        self.position_single = PositionalEncoding(
            d_model=self.dim_model,
            dropout=dropout_tr,
            max_len=self.pad_size,
        )

        enc_layer_single = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=num_head,
            dim_feedforward=forward_hidden,
            dropout=dropout_tr,
            batch_first=True,  # x: (B, T, F)
        )

        # # One TransformerEncoder per channel
        # self.single_encoders = nn.ModuleList(
        #     [
        #         nn.TransformerEncoder(enc_layer_single, num_layers=num_encoder)
        #         for _ in range(num_channels)
        #     ]
        # )

        # Alternative: shared single-channel transformer for all channels
        self.single_encoder = nn.TransformerEncoder(enc_layer_single, num_layers=num_encoder)


        # --- Multichannel fusion transformer block ---
        self.drop_multi = nn.Dropout(p=dropout_tf)
        self.layer_norm_multi = nn.LayerNorm(self.dim_model * num_channels)

        self.position_multi = PositionalEncoding(
            d_model=self.dim_model * num_channels,
            dropout=dropout_tr,
            max_len=self.pad_size,
        )

        enc_layer_multi = nn.TransformerEncoderLayer(
            d_model=self.dim_model * num_channels,
            nhead=num_head,
            dim_feedforward=forward_hidden,
            dropout=dropout_tr,
            batch_first=True,
        )
        self.transformer_encoder_multi = nn.TransformerEncoder(
            enc_layer_multi,
            num_layers=num_encoder_multi,
        )

        # # --- Classifier ---
        # in_fc = self.pad_size * self.dim_model * num_channels
        # self.fc1 = nn.Sequential(
        #     nn.Linear(in_fc, fc_hidden),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_tf),
        # )
        # self.fc2 = nn.Linear(fc_hidden, num_classes)

        in_fc = self.dim_model * num_channels
        self.fc1 = nn.Sequential(
            nn.Linear(in_fc, fc_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_tf),
        )
        self.fc2 = nn.Linear(fc_hidden, num_classes)


    # ==========================================================
    # 1) Raw 1D → time-frequency images
    # ==========================================================
    def raw_to_tf(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L) raw signal per 30s epoch.
        Returns: (B, C, T, F) time-frequency images.
        T = number of time frames (self.pad_size)
        F = number of frequency bins (self.freq_bins)
        """
        B, C, L = x.shape
        assert C == self.num_channels, "Channel count mismatch."
        expected_L = self.epoch_len_sec * self.fs
        assert L == expected_L, f"Expected length {expected_L}, got {L}."

        # Merge batch and channel for STFT
        xc = x.view(B * C, L)  # (B*C, L)

        spec = torch.stft(
            xc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window.to(x.device),
            center=False,
            return_complex=True,
        )  # (B*C, freq_all, T)

        # Magnitude → log-power (dB)
        spec = spec.abs()                      # (B*C, F_all, T)
        spec = 20 * torch.log10(spec + 1e-8)

        # Drop DC (freq=0), keep next self.freq_bins bins
        spec = spec[:, 1 : self.freq_bins + 1, :]   # (B*C, F, T)

        # Per-(sample,channel) normalization over time & freq
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True) + 1e-5
        spec = (spec - mean) / std

        # Reshape to (B, C, T, F)
        spec = spec.view(B, C, self.freq_bins, self.pad_size)  # (B, C, F, T)
        spec = spec.permute(0, 1, 3, 2)                        # (B, C, T, F)

        return spec

    # ==========================================================
    # 2) Forward: raw → TF images → transformer → logits
    # ==========================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L) raw per-epoch signal.
        Returns: logits with shape (B, num_classes).
        """
        # 1) Raw → time-frequency images
        x_img = self.raw_to_tf(x)        # (B, C, T, F)
        B, C, T, F = x_img.shape

        assert C == self.num_channels
        assert T == self.pad_size
        assert F == self.dim_model

        # # 2) Single-channel transformer per channel
        # O_list = []
        # for c in range(C):
        #     xc = x_img[:, c, :, :]               # (B, T, F)
        #     xc = self.position_single(xc)        # add PE
        #     xc = self.single_encoders[c](xc)     # transformer stack
        #     O_list.append(xc)
        
        # Alternative: shared single-channel transformer
        O_list = []
        for c in range(C):
            xc = x_img[:, c, :, :]            # (B, T, F)
            xc = self.position_single(xc)
            xc = self.single_encoder(xc)      # shared for all channels
            O_list.append(xc)

        # 3) Multichannel fusion transformer
        x_multi = torch.cat(O_list, dim=2)       # (B, T, C*F)
        x_multi = self.drop_multi(x_multi)
        # x_multi = self.layer_norm_multi(x_multi)
        residual = x_multi

        x_multi = self.layer_norm_multi(x_multi)
        x_multi = self.position_multi(x_multi)
        x_multi = self.transformer_encoder_multi(x_multi)  # (B, T, C*F)

        # Outer residual
        # x_multi = self.layer_norm_multi(x_multi + residual)
        x_multi = x_multi + residual

        # # 4) Classifier
        # x_flat = x_multi.reshape(B, -1)          # (B, T*C*F)
        # x = self.fc1(x_flat)                     # (B, fc_hidden)
        # logits = self.fc2(x)                     # (B, num_classes)

        x_pooled = x_multi.mean(dim=1)               # (B, C*F) - average pooling over time
        x = self.fc1(x_pooled)                       # (B, fc_hidden)
        logits = self.fc2(x)                         # (B, num_classes)

        return logits
    


class PositionalEncoding_v2(nn.Module):
    """
    Positional encoding following the paper's approach.
    Computes PE once and adds it to the input.
    """
    
    def __init__(self, d_model: int = 128, dropout: float = 0.1, max_len: int = 30):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute PE in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        Returns: (B, T, d_model) with PE added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiChannelSleepNet_v2(nn.Module):
    """
    MultiChannelSleepNet v2: Exact architecture from the paper, but with internal STFT.
    
    Mirrors the original Transformer class but:
    - Takes raw 1D signals (B, C, L) as input
    - Performs STFT internally
    - Uses per-channel transformers (separate encoders for each channel, not shared)
    - Uses flattening classifier (not mean pooling)
    - Follows paper architecture exactly
    """
    
    def __init__(
        self,
        num_channels: int = 2,              # Original paper: 3 (we'll use 2 or 6)
        num_classes: int = 5,
        fs: int = 128,                      # sampling frequency (Hz)
        epoch_len_sec: int = 30,            # each window is 30s
        n_fft: int = 256,                   # FFT size
        dim_model: int = 128,               # feature dimension (freq_bins)
        num_head: int = 4,
        forward_hidden: int = 512,          # FFN hidden dim
        num_encoder: int = 3,               # single-channel transformer depth (layers)
        num_encoder_multi: int = 2,         # multichannel transformer depth (layers)
        fc_hidden: int = 512,               # classifier hidden dim
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # --- STFT / time-frequency parameters ---
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.fs = fs
        self.epoch_len_sec = epoch_len_sec
        self.n_fft = n_fft
        self.dim_model = dim_model
        
        # STFT window & hop (2s window, 1s overlap)
        self.win_length = int(2 * fs)       # samples per window
        self.hop_length = int(1 * fs)       # stride between windows
        
        # Frequency bins after dropping DC: n_fft/2
        self.freq_bins = self.n_fft // 2    # e.g. 128
        assert self.dim_model == self.freq_bins, (
            f"dim_model ({dim_model}) must equal freq_bins ({self.freq_bins}). "
            f"Set dim_model={self.freq_bins} or adjust n_fft."
        )
        
        # Number of time steps (frames) for 30s
        L_expected = epoch_len_sec * fs
        self.pad_size = int((L_expected - self.win_length) / self.hop_length) + 1
        
        # STFT window (Hann)
        self.register_buffer(
            "stft_window",
            torch.hann_window(self.win_length),
            persistent=False,
        )
        
        # --- Single-channel transformer blocks (per-channel, NOT shared) ---
        self.position_single = PositionalEncoding_v2(
            d_model=self.dim_model,
            dropout=dropout,
            max_len=self.pad_size
        )
        
        enc_layer_single = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=num_head,
            dim_feedforward=forward_hidden,
            dropout=dropout,
            batch_first=True,
        )
        
        # One transformer per channel (following paper exactly)
        self.single_encoders = nn.ModuleList([
            nn.TransformerEncoder(enc_layer_single, num_layers=num_encoder)
            for _ in range(num_channels)
        ])
        
        # --- Multichannel fusion transformer block ---
        self.drop_multi = nn.Dropout(p=0.5)  # Paper uses 0.5
        self.layer_norm_multi = nn.LayerNorm(self.dim_model * num_channels)
        
        self.position_multi = PositionalEncoding_v2(
            d_model=self.dim_model * num_channels,
            dropout=dropout,
            max_len=self.pad_size
        )
        
        enc_layer_multi = nn.TransformerEncoderLayer(
            d_model=self.dim_model * num_channels,
            nhead=num_head,
            dim_feedforward=forward_hidden,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder_multi = nn.TransformerEncoder(
            enc_layer_multi,
            num_layers=num_encoder_multi,
        )
        
        # --- Classifier (flattening, like paper) ---
        in_fc = self.pad_size * self.dim_model * num_channels
        self.fc1 = nn.Sequential(
            nn.Linear(in_fc, fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(fc_hidden, num_classes)
    
    def raw_to_tf(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L) raw signal per 30s epoch.
        Returns: (B, C, T, F) time-frequency images.
        """
        B, C, L = x.shape
        assert C == self.num_channels, f"Channel mismatch: expected {self.num_channels}, got {C}"
        expected_L = self.epoch_len_sec * self.fs
        assert L == expected_L, f"Expected length {expected_L}, got {L}"
        
        # Merge batch and channel for STFT
        xc = x.view(B * C, L)  # (B*C, L)
        
        spec = torch.stft(
            xc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window.to(x.device),
            center=False,
            return_complex=True,
        )  # (B*C, freq_all, T)
        
        # Magnitude → log-power (dB)
        spec = spec.abs()
        spec = 20 * torch.log10(spec + 1e-8)
        
        # Drop DC (freq=0), keep next freq_bins bins
        spec = spec[:, 1 : self.freq_bins + 1, :]  # (B*C, F, T)
        
        # Per-(sample,channel) normalization over time & freq
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True) + 1e-5
        spec = (spec - mean) / std
        
        # Reshape to (B, C, T, F)
        spec = spec.view(B, C, self.freq_bins, self.pad_size)  # (B, C, F, T)
        spec = spec.permute(0, 1, 3, 2)  # (B, C, T, F)
        
        return spec
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L) raw per-epoch signal.
        Returns: logits (B, num_classes).
        """
        # 1) Raw → time-frequency images
        x_img = self.raw_to_tf(x)  # (B, C, T, F)
        B, C, T, F = x_img.shape
        
        assert C == self.num_channels
        assert T == self.pad_size
        assert F == self.dim_model
        
        # 2) Single-channel transformer per channel (separate encoders, following paper)
        O_list = []
        for c in range(C):
            xc = x_img[:, c, :, :]  # (B, T, F)
            xc = self.position_single(xc)  # Add PE
            xc = self.single_encoders[c](xc)  # Use per-channel encoder
            O_list.append(xc)
        
        # 3) Concatenate and fuse
        x_multi = torch.cat(O_list, dim=2)  # (B, T, C*F)
        x_multi = self.drop_multi(x_multi)
        x_multi = self.layer_norm_multi(x_multi)
        residual = x_multi
        
        x_multi = self.position_multi(x_multi)
        x_multi = self.transformer_encoder_multi(x_multi)  # (B, T, C*F)
        
        x_multi = self.layer_norm_multi(x_multi + residual)  # Residual + post-norm (like paper)
        
        # 4) Classifier (flatten, like paper)
        x_flat = x_multi.view(B, -1)  # (B, T*C*F)
        x = self.fc1(x_flat)
        logits = self.fc2(x)
        
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
            dropout=0.2,        # adjust manually
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
    


class SSLEpochTransformerConv1D_v2(nn.Module):
    """
    Just a copy of SL version, but no classification head (returning the full transformer output).
    2 options for tokenization (target tokens = 480 or 240).
    """
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
        target_tokens: int = 240,   # 480 or 240
    ):
        super().__init__()
        self.d_model = d_model

        # self.nhead = nhead
        # self.num_layers = num_layers
        # self.dim_feedforward = dim_feedforward

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
    
    def forward(self, x, return_mean_embedding=False):
        x = self.tokenization(x)  # (batch, d_model, target_tokens)
        x = x.transpose(1, 2)   # (batch, target_tokens, d_model)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)  # (batch, target_tokens, d_model)
        mean = x.mean(dim=1)  # mean pooling over tokens, no CLS for now
        if return_mean_embedding:
            return mean  # (batch, d_model), never happens in SSL
        # return whole transformer output for SSL purposes
        return x  # (batch, target_tokens, d_model)




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
        x : torch.Tensor, shape (batch, target_token, d_model)
            Mean of transformer output.
        
        Returns
        -------
        torch.Tensor, shape (batch, num_classes)
            Class logits based on the mean of transformer outputs.
        """
        mean = x.mean(dim=1)  # (batch, d_model)
        logits = self.classifier(mean)  # (batch, num_classes)
        # logits = self.classifier(x)  # (batch, num_classes)
        return logits
    
class SSLLinearProbing(nn.Module):
    """
    Linear Classifier to evaluate the quality of learned SSL embeddings.
    """
    def __init__(
            self,
            d_model=64,
            num_classes=5,
    ):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        """
        Forward pass for linear probing classifier.
        
        Parameters
        ----------
        x : torch.Tensor, shape (batch, target_token, d_model)
            Mean of transformer output.
        
        Returns
        -------
        torch.Tensor, shape (batch, num_classes)
            Class logits based on the mean of transformer outputs.
        """
        mean = x.mean(dim=1)  # (batch, d_model)
        mean_normed = mean / torch.linalg.norm(mean, ord=2, dim=1) # L2 normalization, optional
        logits = self.classifier(mean_normed)  # (batch, num_classes)
        return logits
        