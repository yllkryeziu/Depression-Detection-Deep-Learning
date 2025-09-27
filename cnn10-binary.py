from typing import Optional

import torch
import torch.nn.functional as F

from autrainer.models.abstract_model import AbstractModel
from autrainer.models.utils import ConvBlock, init_bn, init_layer, load_transfer_weights


class Cnn10New(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        feature_extractor_freeze: bool = False,
        dropout_rate: float = 0.5,
        hidden_dim: int = 256,
        in_channels: int = 1,
        transfer: Optional[str] = None,
    ) -> None:
        """CNN10-New model for binary depression classification using mel-spectrograms.
        
        This model uses CNN10 as a feature extractor and adds a classification head
        specifically designed for binary depression detection.

        Args:
            output_dim: Output dimension of the model (should be 2 for binary classification).
            feature_extractor_freeze: Whether to freeze the CNN10 feature extractor weights.
                Defaults to False.
            dropout_rate: Dropout rate for the classification head. Defaults to 0.5.
            hidden_dim: Hidden dimension size for the classification head. Defaults to 256.
            in_channels: Number of input channels. Defaults to 1.
            transfer: Link to the weights to transfer. If None, the weights
                are randomly initialized. Defaults to None.
        """
        super().__init__(output_dim)
        self.feature_extractor_freeze = feature_extractor_freeze
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.transfer = transfer
        
        # CNN10 feature extractor layers
        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        
        # Feature processing layer (from original CNN10)
        self.fc1 = torch.nn.Linear(512, 512, bias=True)
        
        # Custom classification head for depression detection
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(512, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(self.hidden_dim // 2, output_dim)
        )

        self.init_weight()
        
        # Freeze feature extractor if requested
        if self.feature_extractor_freeze:
            self._freeze_feature_extractor()
            
        if self.transfer:  # pragma: no cover
            load_transfer_weights(self, self.transfer)

    def init_weight(self) -> None:
        """Initialize model weights."""
        init_bn(self.bn0)
        init_layer(self.fc1)
        
        # Initialize classification head
        for layer in self.classification_head:
            if isinstance(layer, torch.nn.Linear):
                init_layer(layer)
    
    def _freeze_feature_extractor(self) -> None:
        """Freeze the CNN10 feature extractor weights."""
        for param in self.bn0.parameters():
            param.requires_grad = False
        for param in self.conv_block1.parameters():
            param.requires_grad = False
        for param in self.conv_block2.parameters():
            param.requires_grad = False
        for param in self.conv_block3.parameters():
            param.requires_grad = False
        for param in self.conv_block4.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False

    def extract_cnn10_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using CNN10 architecture.
        
        Args:
            x: Input mel-spectrogram tensor.
            
        Returns:
            Extracted feature embeddings.
        """
        # CNN10 feature extraction pipeline
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        # Clipwise aggregation (global pooling)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        return x

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input features.
        
        Args:
            features: Input mel-spectrogram features.
            
        Returns:
            Feature embeddings from CNN10 feature extractor.
        """
        return self.extract_cnn10_features(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            features: Input mel-spectrogram features.
            
        Returns:
            Classification logits for binary depression detection.
        """
        # Extract features using CNN10
        x = self.embeddings(features)
        
        # Apply classification head for depression detection
        x = self.classification_head(x)
        
        return x 