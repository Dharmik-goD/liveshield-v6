from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class PADResult:
    configured: bool
    score: float | None
    label: str
    probabilities: dict[str, float] | None = None


class PADEngine(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray, face_box: tuple) -> PADResult:
        """Return an experimental model estimate for one detected face."""
        raise NotImplementedError


class UnconfiguredPADEngine(PADEngine):
    def predict(self, image: np.ndarray, face_box: tuple) -> PADResult:
        return PADResult(
            configured=False,
            score=None,
            label="INCONCLUSIVE",
            probabilities=None,
        )
