"""MULTIMOORA with CRITIC weighting package."""

from multimoora_critic.models import MultimooraResult
from multimoora_critic.solver import multimoora_critic

__version__ = "0.1.0"

__all__ = [
    "multimoora_critic",
    "MultimooraResult",
    "__version__",
]
