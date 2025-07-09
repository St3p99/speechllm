"""
SpeechLLM package for speech-to-text and speech-language model tasks.
"""

from .dataset import SpeechDataset
from .model import SpeechLLM, SpeechLLMConfig

__version__ = "0.1.0"

__all__ = [
    "SpeechDataset",
    "SpeechLLM",
] 