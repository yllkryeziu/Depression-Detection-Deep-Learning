import torch
import torch.nn as nn
from typing import Optional
from autrainer.models.abstract_model import AbstractModel


class LSTM(AbstractModel):
    """
    CNN-LSTM model for depression detection using pre-extracted CNN features.
    
    This model consists of:
    1. Bidirectional LSTM sequence processor for temporal patterns
    2. Classification head for final prediction
    
    The CNN feature extraction is done separately and features are loaded as input.
    """
    
    def __init__(
        self,
        output_dim: int,
        feature_dim: int = 512,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        **kwargs
    ):
        """
        Initialize CNN-LSTM Depression model.
        
        Args:
            output_dim: Number of output classes (2 for binary depression classification)
            feature_dim: Dimension of input CNN features (512 for CNN10)
            lstm_hidden_size: Hidden size of LSTM layers
            lstm_num_layers: Number of LSTM layers (2 or 3 recommended)
            lstm_dropout: Dropout rate between LSTM layers
        """
        super().__init__(output_dim)
        
        self.feature_dim = feature_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        
        # Bidirectional LSTM Sequence Processor
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Crucial for using both past and future context
        )
        
        # Calculate the size of concatenated final hidden states
        # (forward + backward hidden states)
        final_hidden_size = lstm_hidden_size * 2  # *2 because bidirectional
        
        # Classification Head
        self.classifier = nn.Linear(final_hidden_size, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN-LSTM model.
        
        Args:
            features: Pre-extracted CNN features with shape 
                     (batch_size, sequence_length, feature_dim)
        
        Returns:
            Logit scores for each class with shape (batch_size, output_dim)
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # Bidirectional LSTM processing
        # lstm_out: (batch_size, seq_len, hidden_size * 2)
        # hidden: (num_layers * 2, batch_size, hidden_size) - final hidden states
        # cell: (num_layers * 2, batch_size, hidden_size) - final cell states
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Extract final hidden states from forward and backward directions
        # hidden shape: (num_layers * 2, batch_size, hidden_size)
        # We want the last layer's hidden states
        forward_hidden = hidden[-2]  # Forward direction of last layer
        backward_hidden = hidden[-1]  # Backward direction of last layer
        
        # Concatenate forward and backward final hidden states
        # Shape: (batch_size, hidden_size * 2)
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Classification head
        # Output shape: (batch_size, output_dim)
        logits = self.classifier(final_hidden)
        
        return logits
    
    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings (final hidden states before classification).
        
        Args:
            features: Pre-extracted CNN features with shape 
                     (batch_size, sequence_length, feature_dim)
        
        Returns:
            Final hidden state embeddings with shape (batch_size, hidden_size * 2)
        """
        batch_size, seq_len, feature_dim = features.shape
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Extract and concatenate final hidden states
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return final_hidden 