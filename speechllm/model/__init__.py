from .modeling_speechllm import SpeechLLM
from .configuration_speechllm import SpeechLLMConfig
from .speech_encoder import WavLMEncoder, HubertEncoder
from .projector import MLPProjector
from .text_decoder import LlamaDecoder
from .downsample import Downsample

# Register with transformers AutoModel system
from transformers import AutoConfig, AutoModel

# Register config and model
AutoConfig.register("speechllm", SpeechLLMConfig)
AutoModel.register(SpeechLLMConfig, SpeechLLM)

__all__ = [
    "SpeechLLM",
    "SpeechLLMConfig",
    "WavLMEncoder", 
    "HubertEncoder",
    "MLPProjector",
    "LlamaDecoder",
    "Downsample",
]
