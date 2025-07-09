import contextlib
import logging
from abc import ABCMeta, abstractmethod
from typing import List, Optional

# import deepspeed
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaModel,
    PreTrainedModel,
)
# from transformers.integrations import (
#     deepspeed_config,
#     is_deepspeed_zero3_enabled,
# )

from speechllm import conversation as conversation_lib


logger = logging.getLogger(__name__)


class HfTextDecoder(
    torch.nn.Module, metaclass=ABCMeta
):
    """
    Base class for Hugging Face text decoders. This class is a wrapper
    around Hugging Face text decoders that provides a consistent interface for
    decoding multimodal embeddings into text.

    Args:
        name_or_path (str, optional): The model name or path to load the model
          from. If not provided, it will be inferred from `config_kwargs`.
        config_dict: Keyword arguments used to override the default values
          in the model configuration. Useful for loading pre-trained models
          with a configuration file attached to them.
        attn_implementation (str, optional): The attention implementation to
          use. If None, the default attention implementation from the model
          configuration is used. Defaults to None.
        torch_dtype (torch.dtype, optional): The data type to use for the
          model. If None, the default data type from the model configuration
          is used. Defaults to None.
        cache_dir (str, optional): The directory to cache pretrained weights.
    """

    config_class = AutoConfig
    tokenizer_class = AutoTokenizer
    tokenizer_text_arg_name = None
    model_class = None
    model_for_causal_lm_class = None
    model_forward_kwargs = {}

    def __init__(
        self,
        name_or_path: Optional[str] = None,
        config_dict: Optional[dict] = None,
        add_lm_head: bool = True,
        tokenizer_padding_side: str = "right",
        conversation_version: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        allow_hf_hub: bool = True,
    ):
        if self.model_class is None or not issubclass(
            self.model_class, PreTrainedModel
        ):
            raise ValueError(
                f"Class attribute `model_class` must be a subclass of "
                f"`transformers.PreTrainedModel` (found {self.model_class})."
            )

        self._supports_flash_attn_2 = self.model_class._supports_flash_attn_2
        self._supports_sdpa = self.model_class._supports_sdpa

        # NOTE: we call `super().__init__` now because
        # `AttentionImplementationMixin` expects to find the
        # `_supports_flash_attn_2` and `_supports_sdpa` attributes
        super().__init__()

        self.set_attn_implementation_with_fallback(attn_implementation)

        if add_lm_head and (
            self.model_for_causal_lm_class is None
            or not issubclass(self.model_for_causal_lm_class, PreTrainedModel)
        ):
            raise ValueError(
                f"Class attribute `model_for_causal_lm_class` must be a "
                f"subclass of `transformers.PreTrainedModel` (found "
                f"{self.model_for_causal_lm_class})."
            )
        self.add_lm_head = add_lm_head

        config_dict = config_dict or {}
        self.name_or_path = name_or_path or config_dict.pop(
            "_name_or_path", None
        )
        if self.name_or_path is None:
            raise ValueError(
                "`name_or_path` must be provided either as an explicit "
                "argument or as part of `config_kwargs` (in which case it "
                "should be named `_name_or_path`). Without it, how would you "
                "initialize the tokenizer?"
            )

        torch_dtype_in_config = config_dict.pop("torch_dtype", None)
        torch_dtype = torch_dtype or torch_dtype_in_config
        self.config = self.config_class.from_pretrained(
            self.name_or_path, torch_dtype=torch_dtype, **config_dict
        )

        self.tokenizer = self.tokenizer_class.from_pretrained(
            self.name_or_path,
            cache_dir=cache_dir,
            padding_side=tokenizer_padding_side,
        )

        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.allow_hf_hub = allow_hf_hub

        target_model_class = self.model_class
        if self.add_lm_head:
            target_model_class = self.model_for_causal_lm_class

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.torch_dtype or default_dtype)

        # Initialize model without deepspeed
        self.model = target_model_class(self.config)

        torch.set_default_dtype(default_dtype)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Hook this wrapper's config to the model's config so that each
        # update to `self.config` is reflected in the model's config and
        # vice versa
        self.config = self.model.config

        if conversation_version is None:
            raise ValueError("`conversation_version` must be provided.")
        self.conversation_version = conversation_version
        self._adapt_tokenizer_to_conversation_version()

    def set_attn_implementation_with_fallback(self, attn_implementation):
        """Set attention implementation with fallback logic."""
        if attn_implementation is None:
            # Use default implementation
            return
        
        # Simple fallback logic - in a real implementation this would be more sophisticated
        if attn_implementation == "flash_attention_2" and not self._supports_flash_attn_2:
            logging.warning("flash_attention_2 not supported, falling back to default")
            return
        elif attn_implementation == "sdpa" and not self._supports_sdpa:
            logging.warning("sdpa not supported, falling back to default")
            return
        
        # Set the attention implementation if supported
        if hasattr(self, '_attn_implementation'):
            self._attn_implementation = attn_implementation

    @property
    @abstractmethod
    def _default_conversation_version(self):
        pass

    def _adapt_tokenizer_to_conversation_version(self):
        if self.conversation_version == "v0":
            if self.tokenizer.pad_token is None:
                pad_token = "[PAD]"
                logger.info(f"Adding pad token '{pad_token}'.")
                self._resize_tokenizer_and_embedding_layer(
                    additional_special_tokens=[pad_token],
                )
        elif self.conversation_version == "v0.5":
            self.tokenizer.pad_token = self.tokenizer.unk_token
        elif self.conversation_version == "v1":
            self.tokenizer.pad_token = self.tokenizer.unk_token
            if self.conversation_version in conversation_lib.conv_templates:
                conversation_lib.default_conversation = (
                    conversation_lib.conv_templates[self.conversation_version]
                )
            else:
                conversation_lib.default_conversation = (
                    conversation_lib.conv_templates["vicuna_v1"]
                )
        elif self.conversation_version == "mistral_instruct":
            self.tokenizer.pad_token = self.tokenizer.unk_token
            conversation_lib.default_conversation = (
                conversation_lib.conv_templates["mistral_instruct"]
            )
        elif (
            self.conversation_version == "llama_3_1"
            or self.conversation_version == "llama_3_1_base"
            or self.conversation_version == "llama_3_1_lara"
        ):
            pad_token = "<|finetune_right_pad_id|>"
            logger.info(f"Setting pad token to '{pad_token}'.")
            self.tokenizer.pad_token = pad_token
            self.tokenizer.pad_token_id = 128004
            self.model.generation_config.eos_token_id = 128009
            # ↑ 128009 = <|eot_id|>
            conversation_lib.default_conversation = (
                conversation_lib.conv_templates[self.conversation_version]
            )
        else:
            if (
                self.conversation_version
                not in conversation_lib.conv_templates
            ):
                raise ValueError(
                    f"Unknown conversation format "
                    f"'{self.conversation_version}'."
                )
            conversation_lib.default_conversation = (
                conversation_lib.conv_templates[self.conversation_version]
            )
            if self.tokenizer.pad_token is None:
                pad_token = "<pad>"
                logger.info(f"Adding pad token as {pad_token}")
                self._resize_tokenizer_and_embedding_layer(
                    additional_special_tokens=[pad_token],
                )

        logger.info(
            f"Using conversation format: "
            f"{conversation_lib.default_conversation.version}"
        )

    def _resize_tokenizer_and_embedding_layer(
        self,
        additional_special_tokens: List[str],
    ):
        """
        Note: This is the unoptimized version that may make your embedding size
        not be divisible by 64.
        """
        special_tokens_dict = {
            "additional_special_tokens": additional_special_tokens
        }
        num_new_tokens = self.tokenizer.add_special_tokens(
            special_tokens_dict, replace_additional_special_tokens=False
        )
        if num_new_tokens > 0:  # i.e. if they were not already there
            self.model.resize_token_embeddings(
                new_num_tokens=len(self.tokenizer)
            )
            input_embeddings = self.model.get_input_embeddings().weight.data
            mean_input_embedding = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            input_embeddings[-num_new_tokens:] = mean_input_embedding

            output_embeddings = self.model.get_output_embeddings().weight.data
            mean_output_embedding = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings[-num_new_tokens:] = mean_output_embedding

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    @property
    def hidden_size(self):
        return self.config.hidden_size
    
class LlamaDecoder(HfTextDecoder):
    tokenizer_class = AutoTokenizer
    tokenizer_text_arg_name = "text"
    model_class = LlamaModel
    model_for_causal_lm_class = LlamaForCausalLM

    def __init__(
        self,
        name_or_path: Optional[str] = None,
        config_dict: Optional[dict] = None,
        add_lm_head: bool = True,
        tokenizer_padding_side: str = "right",
        conversation_version: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(
            name_or_path=name_or_path,
            config_dict=config_dict,
            add_lm_head=add_lm_head,
            tokenizer_padding_side=tokenizer_padding_side,
            conversation_version=conversation_version,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )

        if "Llama-3" in self.model.config._name_or_path:
            if not "Llama-3." in self.model.config._name_or_path:
                raise ValueError("Only Llama 3.X models are supported.")

            # https://llama.com/docs/model-cards-and-prompt-formats/llama3_1
            pad_token, pad_token_id = "<|finetune_right_pad_id|>", 128004
            self.tokenizer.pad_token = pad_token
            self.tokenizer.pad_token_id = pad_token_id
            self.config.pad_token_id = pad_token_id

    @property
    def _default_conversation_version(self):
        return "llama_3_1"