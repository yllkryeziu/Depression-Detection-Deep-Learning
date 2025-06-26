import torch
import torchaudio
from autrainer.transforms import AbstractTransform


class MFCC(AbstractTransform):
    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int,
        dct_type: int,
        norm: str,
        log_mels: bool,
        melkwargs: dict | None = None,
        order: int = -90,
    ) -> None:
        super().__init__(order)
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.log_mels = log_mels
        self.melkwargs = melkwargs
        self._mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            dct_type=self.dct_type,
            norm=self.norm,
            log_mels=self.log_mels,
            melkwargs=self.melkwargs,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # shape: (1, time, n_mfcc)
        return self._mfcc(x).permute(0, 2, 1)
