from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from autrainer.models.abstract_model import AbstractModel

__version__ = "0.1.0"

class CNNGRUModel(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        cnn_model: str = "vgg16",
        pretrained: bool = True,
        feature_dim: int = 4096,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.1,
        fc_hidden_size: int = 32,
        fc_dropout: float = 0.1,
        freeze_cnn: bool = False,
    ):
        super().__init__(output_dim)
        
        self.cnn_model = cnn_model
        self.pretrained = pretrained
        self.feature_dim = feature_dim
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.gru_dropout = gru_dropout
        self.fc_hidden_size = fc_hidden_size
        self.fc_dropout = fc_dropout
        self.freeze_cnn = freeze_cnn
        
        self.cnn = self._create_cnn_backbone()
        
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            dropout=self.gru_dropout if self.gru_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.gru_hidden_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_hidden_size, output_dim)
        )
        
        self.feature_norm = nn.BatchNorm1d(self.feature_dim)
        
    def _create_cnn_backbone(self):
        if self.cnn_model == "vgg16":
            model = models.vgg16(pretrained=self.pretrained)
            # Extract features from second FC layer (fc2)
            # VGG16 classifier: fc1(25088->4096) -> relu -> dropout -> fc2(4096->4096) -> relu -> dropout -> fc3(4096->1000)
            feature_extractor = nn.Sequential(
                model.features,  # Convolutional layers
                model.avgpool,   # Adaptive average pooling
                nn.Flatten(),
                model.classifier[0],  # fc1: 25088 -> 4096
                model.classifier[1],  # ReLU
                model.classifier[2],  # Dropout
                model.classifier[3],  # fc2: 4096 -> 4096
                model.classifier[4],  # ReLU
            )
            self.feature_dim = 4096
            
        elif self.cnn_model == "alexnet":
            model = models.alexnet(pretrained=self.pretrained)
            # Extract features from second FC layer
            feature_extractor = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten(),
                model.classifier[0],  # fc1: 9216 -> 4096
                model.classifier[1],  # ReLU
                model.classifier[2],  # Dropout
                model.classifier[3],  # fc2: 4096 -> 4096
                model.classifier[4],  # ReLU
            )
            self.feature_dim = 4096
            
        elif self.cnn_model == "densenet121":
            model = models.densenet121(pretrained=self.pretrained)
            feature_extractor = nn.Sequential(
                model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.feature_dim = 1024
            
        elif self.cnn_model == "densenet201":
            model = models.densenet201(pretrained=self.pretrained)
            feature_extractor = nn.Sequential(
                model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.feature_dim = 1920
            
        else:
            raise ValueError(f"Unsupported CNN model: {self.cnn_model}")
        
        if self.freeze_cnn:
            for param in feature_extractor.parameters():
                param.requires_grad = False
                
        return feature_extractor
    
    def _convert_to_rgb(self, x):
        if x.shape[-3] == 1:  # Single channel
            x = x.repeat(*([1] * (len(x.shape) - 3)), 3, 1, 1)
        return x

    def extract_cnn_features(self, mel_spectrograms):
        """
            mel_spectrograms: Batch of mel spectrograms [batch_size, seq_len, channels, height, width]
        """
        batch_size, seq_len = mel_spectrograms.shape[:2]
        
        mel_flat = mel_spectrograms.view(-1, *mel_spectrograms.shape[2:])
        
        # Convert single channel to RGB for pretrained models
        mel_flat = self._convert_to_rgb(mel_flat)
        
        cnn_features = self.cnn(mel_flat)
        
        if cnn_features.shape[0] > 1:
            cnn_features = self.feature_norm(cnn_features)
        
        cnn_features = cnn_features.view(batch_size, seq_len, self.feature_dim)
        
        return cnn_features
    
    def forward(self, features):
        if features.dtype == torch.uint8:
            features = features.float() / 255.0
        
        if len(features.shape) == 4:
            features = features.unsqueeze(1)
        
        cnn_features = self.extract_cnn_features(features)
        
        gru_output, _ = self.gru(cnn_features)
        
        last_output = gru_output[:, -1, :]

        logits = self.fc(last_output)
        
        return logits
    
    def embeddings(self, features):
        if features.dtype == torch.uint8:
            features = features.float() / 255.0
            
        if len(features.shape) == 4:
            features = features.unsqueeze(1)
        
        cnn_features = self.extract_cnn_features(features)
        
        gru_output, _ = self.gru(cnn_features)
        
        return gru_output[:, -1, :]


class VGG16GRUModel(CNNGRUModel):
    def __init__(self, output_dim: int, pretrained: bool = True, **kwargs):
        super().__init__(
            output_dim=output_dim,
            cnn_model="vgg16",
            pretrained=pretrained,
            feature_dim=4096,
            **kwargs
        )


class AlexNetGRUModel(CNNGRUModel):
    def __init__(self, output_dim: int, pretrained: bool = True, **kwargs):
        super().__init__(
            output_dim=output_dim,
            cnn_model="alexnet",
            pretrained=pretrained,
            feature_dim=4096,
            **kwargs
        )


class DenseNet121GRUModel(CNNGRUModel):
    def __init__(self, output_dim: int, pretrained: bool = True, **kwargs):
        super().__init__(
            output_dim=output_dim,
            cnn_model="densenet121",
            pretrained=pretrained,
            feature_dim=1024,
            **kwargs
        )


class DenseNet201GRUModel(CNNGRUModel):
    def __init__(self, output_dim: int, pretrained: bool = True, **kwargs):
        super().__init__(
            output_dim=output_dim,
            cnn_model="densenet201",
            pretrained=pretrained,
            feature_dim=1920,
            **kwargs
        ) 