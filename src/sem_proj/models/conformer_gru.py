import torch
import torch.nn as nn

class ConformerFeatureExtractor(nn.Module):
    """
    Wraps your Conformer’s PatchEmbedding + TransformerEncoder
    and returns either:
      - tokens: (B, N, E)
      - pooled vector per epoch: (B, E)
    """
    def __init__(self, conformer, pool: str = "mean"):
        super().__init__()
        # reuse already-created modules
        self.patch_embedding = conformer[0]
        self.encoder = conformer[1]
        self.pool = pool

    def forward(self, x, return_tokens: bool = False):
        """
        x: (B, 1, C, T)
        returns:
          - if return_tokens: (B, N, E)
          - else: (B, E) or (B, N*E) depending on pool
        """
        # (B, 1, C, T) -> (B, N, E)
        tokens = self.patch_embedding(x)
        tokens = self.encoder(tokens) # still (B, N, E)

        if return_tokens:
            return tokens # (B, N, E)
        
        if self.pool == "mean":
            # global average over tokens: (B, N, E) -> (B, E)
            return tokens.mean(dim=1)
        
        if self.pool == "flatten":
            # flatten all tokens: (B, N, E) -> (B, N*E)
            return tokens.contiguous().view(tokens.size(0), -1)
        
        # no pooling → just return tokens
        return tokens # (B, N, E)
    

class ConformerGRUClassifier(nn.Module):
    """
    Conformer feature extractor + GRU + classification head
    """
    def __init__(self, backbone: ConformerFeatureExtractor,
                 emb_dim: int,
                 gru_hidden: int,
                 num_layers: int,
                 num_classes: int):
        super().__init__()
        self.backbone = backbone # returns (B, emb_dim) per epoch
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.classifier = nn.Linear(gru_hidden, num_classes)

    # training / offline
    def forward(self, x_seq):
        """
        x_seq: (B, T_epochs, 1, C, T_samples)
        returns: logits (B, T_epochs, num_classes)
        """
        B, T_epochs, _, C, T_samples = x_seq.shape

        # merge batch and time, process all epochs with Conformer
        x_flat = x_seq.view(B * T_epochs, 1, C, T_samples)
        h_flat = self.backbone(x_flat)  # (B*T_epochs, emb_dim)

        # restore sequence shape
        emb_dim = h_flat.size(-1)
        h_seq = h_flat.view(B, T_epochs, emb_dim)  # (B, T_epochs, emb_dim)

        # GRU over epochs
        gru_out, _ = self.gru(h_seq)    # (B, T_epochs, gru_hidden)

        # per-epoch classification (sleep stage)
        logits = self.classifier(gru_out)  # (B, T_epochs, num_classes)
        return logits
    
    # real-time
    def forward_step(self, x_epoch, h_gru=None):
        # x_epoch: (B, 1, C, T_samples)
        z = self.backbone(x_epoch)  # (B, emb_dim)
        gru_out, h_new = self.gru(z.unsqueeze(1), h_gru)
        logits = self.classifier(gru_out[:, -1, :])  # (B, num_classes)
        return logits, h_new
    
