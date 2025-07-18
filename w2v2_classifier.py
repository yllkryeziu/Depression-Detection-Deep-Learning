import warnings
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from autrainer.models.abstract_model import AbstractModel


class W2V2Backbone(nn.Module):
    def __init__(
        self,
        model_name,
        freeze_extractor: bool = True,
        time_pooling: bool = True,
        transfer: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.freeze_extractor = freeze_extractor
        self.time_pooling = time_pooling
        self.transfer = transfer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if transfer:  # pragma: no cover
                model = Wav2Vec2Model.from_pretrained(self.model_name)
            else:
                config = Wav2Vec2Config.from_pretrained(
                    self.model_name,
                    output_hidden_states=True,
                    return_dict=True,
                )
                model = Wav2Vec2Model(config)
        self.model = model
        self.output_dim = model.config.hidden_size
        if self.freeze_extractor:
            self.model.freeze_feature_encoder()

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        features = self.model(features)["last_hidden_state"]
        if self.time_pooling:
            features = features.mean(1)
        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.embeddings(features)


class ClassificationHead(nn.Module):
    """Classification head for depression detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class W2V2Classifier(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        model_name: str,
        freeze_extractor: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        transfer: bool = False,
    ) -> None:
        """Wave2Vec2 model with classification head for depression detection.
        
        Args:
            output_dim: Output dimension (number of classes, typically 2 for depression detection).
            model_name: Name of the Wave2Vec2 model loaded from Huggingface.
            freeze_extractor: Whether to freeze the feature extractor.
            hidden_dim: Hidden dimension of the classification head.
            dropout: Dropout rate. Defaults to 0.5.
            transfer: Whether to initialize the Wave2Vec2 backbone with
                pretrained weights. Defaults to False.
        """
        super().__init__(output_dim)
        self.model_name = model_name
        self.freeze_extractor = freeze_extractor
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.transfer = transfer
        
        # Initialize Wave2Vec2 backbone
        self.backbone = W2V2Backbone(
            model_name=model_name,
            freeze_extractor=freeze_extractor,
            time_pooling=True,
            transfer=transfer,
        )
        
        # Initialize classification head
        self.classification_head = ClassificationHead(
            input_dim=self.backbone.output_dim,
            hidden_dim=hidden_dim,
            num_classes=output_dim,
            dropout=dropout,
        )

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from Wave2Vec2 backbone."""
        return self.backbone(features.squeeze(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classification head."""
        embeddings = self.embeddings(features)
        return self.classification_head(embeddings)


class W2V2ClassifierFrozen(W2V2Classifier):
    """Wave2Vec2 classifier with frozen feature extractor."""
    
    def __init__(
        self,
        output_dim: int,
        model_name: str,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        transfer: bool = False,
    ) -> None:
        """Wave2Vec2 classifier with frozen feature extractor.
        
        Args:
            output_dim: Output dimension (number of classes).
            model_name: Name of the Wave2Vec2 model.
            hidden_dim: Hidden dimension of the classification head.
            dropout: Dropout rate.
            transfer: Whether to use pretrained weights.
        """
        super().__init__(
            output_dim=output_dim,
            model_name=model_name,
            freeze_extractor=True,
            hidden_dim=hidden_dim,
            dropout=dropout,
            transfer=transfer,
        )


class W2V2ClassifierFineTuned(W2V2Classifier):
    """Wave2Vec2 classifier with fine-tunable feature extractor."""
    
    def __init__(
        self,
        output_dim: int,
        model_name: str,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        transfer: bool = False,
    ) -> None:
        """Wave2Vec2 classifier with fine-tunable feature extractor.
        
        Args:
            output_dim: Output dimension (number of classes).
            model_name: Name of the Wave2Vec2 model.
            hidden_dim: Hidden dimension of the classification head.
            dropout: Dropout rate.
            transfer: Whether to use pretrained weights.
        """
        super().__init__(
            output_dim=output_dim,
            model_name=model_name,
            freeze_extractor=False,
            hidden_dim=hidden_dim,
            dropout=dropout,
            transfer=transfer,
        ) 