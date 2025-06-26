import torch
from autrainer.models import AbstractModel
from torch.nn import GRU, Conv2d, Linear, MaxPool2d, ReLU

__version__ = "0.1.0"


class CRNN(AbstractModel):
    def __init__(
        self,
        output_dim: int,
        hidden_size: int,
        gru_layers: int,
    ) -> None:
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        super().__init__(output_dim)

        self.conv1 = Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = Conv2d(64, self.hidden_size, kernel_size=3, padding=1)
        self.pool = MaxPool2d(kernel_size=(1, 2))
        self.relu = ReLU()
        self.gru = GRU(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = Linear(2 * self.hidden_size, self.output_dim)

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        for conv in [self.conv1, self.conv2, self.conv3]:
            x = self.relu(conv(x))
            x = self.pool(x)
        x = x.mean(dim=-1).permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        return self.fc(x)
