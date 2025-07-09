import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from transformers import AutoConfig, PretrainedConfig

logger = logging.getLogger(__name__)

class SpeechLLMConfig(PretrainedConfig):
    model_type = "speechllm"
    is_composition = True

    def __init__(
        self,
        text_decoder_name_or_path: str = None,
        speech_encoder_name_or_path: str = None,
        freeze_modules: List[str] = None,
        tokenizer_padding_side: str = "right",
        conversation_version: str = None,
        downsample_factor: int = 1,
        projector_n_layers: int = 4,
        projector_activation: str = "relu",
        projector_hidden_size: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store the paths as attributes
        self.text_decoder_name_or_path = text_decoder_name_or_path or kwargs.pop("text_decoder_name_or_path", None)
        self.speech_encoder_name_or_path = speech_encoder_name_or_path or kwargs.pop("speech_encoder_name_or_path", None)
        self.freeze_modules = freeze_modules or kwargs.pop("freeze_modules", [])

        if self.text_decoder_name_or_path is None:
            raise ValueError("`text_decoder_name_or_path` must be provided.")
        
        if self.speech_encoder_name_or_path is None:
            raise ValueError("`speech_encoder_name_or_path` must be provided.")
        
        # Load sub-configs
        self.text_decoder = AutoConfig.from_pretrained(self.text_decoder_name_or_path)
        self.speech_encoder = AutoConfig.from_pretrained(self.speech_encoder_name_or_path)

        # FIXME: added this to please DeepSpeed, but is this
        # the right way to do it? Cause technically a `SpeechLLMModel` has
        # multiple hidden sizes (encoder, adapter, decoder). I guess
        # what matters is the dimensionality of the *output* embeddings?
        self.hidden_size = self.text_decoder.hidden_size
        
        self.conversation_version = conversation_version
        self.tokenizer_padding_side = tokenizer_padding_side
        self.downsample_factor = downsample_factor
        self.projector_n_layers = projector_n_layers
        self.projector_activation = projector_activation
        self.projector_hidden_size = projector_hidden_size
        
        # Allow overriding tie_word_embeddings to fix shared memory warnings
        self.tie_word_embeddings = self.text_decoder.tie_word_embeddings

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        """
        output = super().to_dict()
        
        # Make sure all our custom attributes are included
        output.update({
            "text_decoder_name_or_path": self.text_decoder_name_or_path,
            "speech_encoder_name_or_path": self.speech_encoder_name_or_path,
            "freeze_modules": self.freeze_modules,
            "conversation_version": self.conversation_version,
            "tokenizer_padding_side": self.tokenizer_padding_side,
            "downsample_factor": self.downsample_factor,
            "projector_n_layers": self.projector_n_layers,
            "projector_activation": self.projector_activation,
            "projector_hidden_size": self.projector_hidden_size,
        })
        
        return output