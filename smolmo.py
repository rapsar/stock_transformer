# smol models
import torch.nn as nn
import torch

class TimeSeriesTransformer(nn.Module):
    """
    minimalist transformer (encoder) for time series prediction
    """
    # requires __init__(self, ...) and forward(self, x)
    def __init__(
        self, 
        num_features: int, 
        d_model: int, 
        input_length: int, 
        nhead: int, 
        dim_feedforward: int, 
        num_layers: int, 
        output_length: int,
        device: torch.device = torch.device('cpu')
        ) -> None:
        
        # inherits
        super().__init__()
        
        # device
        self.device = device
        # imbeddings using linear layer
        self.to_embeddings = nn.Linear(in_features = num_features, out_features = d_model, device = device)
        # positional embedding using learnable
        self.pos_embeddings = nn.Parameter(torch.zeros(1, input_length, d_model))
        # defining encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            device = device,
            batch_first = True
            )
        # building stack of layers
        self.encoder = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers = num_layers)
        # unembeddings as linear
        self.fc_out = nn.Linear(in_features = d_model, out_features = output_length, device = device)

    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src shape: (batch_size, input_length, num_features)
        returns: (batch_size, output_length)
        """
        src = src.to(self.device)
        x = self.to_embeddings(src) + self.pos_embeddings
        y = self.encoder(x)
        out = self.fc_out(y[:, -1, :])
        return out