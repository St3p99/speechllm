import argparse
import logging
import os
from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.multiprocessing
import transformers
import yaml

from speechllm.arguments import DataArguments, TrainingArguments
from speechllm.dataset.speech_dataset import SpeechDataset
from speechllm.dataset.utils import DataCollatorForSupervisedDataset

from speechllm.model.configuration_speechllm import SpeechLLMConfig
from speechllm.model.modeling_speechllm import SpeechLLM

from torch.utils.data import DataLoader

from transformers import Trainer

logger = logging.getLogger(__name__)


class SpeechLLMTrainer(Trainer):
    """Custom trainer that preserves audio data in the dataset."""
    
    def _remove_unused_columns(self, dataset, description=None):
        """Override to preserve audio and audio_sr columns."""
        # Don't remove any columns for speech datasets - preserve all data
        return dataset    
    def save_model(self, output_dir=None, _internal_call=False):
        """Override to handle shared memory warnings gracefully."""
        try:
            print(f"Saving model to {output_dir}")
            super().save_model(output_dir, _internal_call)
            if self.model and hasattr(self.model, 'config'):
                print(f"Saving model config to {output_dir}/config.json")
                self.model.config.save_pretrained(output_dir)
        except Exception as e:
            if "share memory" in str(e).lower() or "tied" in str(e).lower():
                logger.warning(f"Encountered shared memory issue during save: {e}")
                logger.info("Attempting to save with safe_serialization=False")
                # Try saving with safe_serialization disabled
                original_safe = getattr(self.args, 'save_safetensors', True)
                self.args.save_safetensors = False
                try:
                    super().save_model(output_dir, _internal_call)
                    if self.model and hasattr(self.model, 'config'):
                        print(f"Saving model config to {output_dir}/config.json")
                        self.model.config.save_pretrained(output_dir)
                    logger.info("Successfully saved model with safe_serialization=False")
                finally:
                    self.args.save_safetensors = original_safe
            else:
                raise e

    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        dataloader.drop_last = True
        return dataloader

def count_parameters(
    module: torch.nn.Module, trainable_only: bool = False
) -> int:
    # if the module parameters have been partitioned into multiple GPUs
    # by DeepSpeed ZeRO-3, we must get the number of elements of each
    # parameter from the `ds_numel` attribute
    return sum(
        getattr(param, "ds_numel", param.numel())
        for param in module.parameters()
        if not trainable_only or param.requires_grad
    )

def count_trainable_parameters(module: torch.nn.Module) -> int:
    return count_parameters(module, trainable_only=True)

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SpeechDataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

def train(path_to_yaml_config: Union[str, Path]):
    with open(path_to_yaml_config, "r") as config_file:
        config_dict = yaml.safe_load(config_file)

    data_args = DataArguments(**config_dict["data"])
    training_args = TrainingArguments(**config_dict["training"])

    if isinstance(training_args.learning_rate, str):
        training_args.learning_rate = float(training_args.learning_rate)

    if training_args.fp16:
        torch_dtype = torch.float16
    elif training_args.bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model_configs = config_dict["model"]
    model_configs["freeze_modules"] = training_args.freeze_modules
    # Create model directly with the specified parameters
    model = SpeechLLM(
        config=SpeechLLMConfig(**model_configs)
    )

    data_module = make_supervised_data_module(
        tokenizer=model.text_decoder.tokenizer, data_args=data_args
    )

    logger.info(f"Total parameters: {count_parameters(model)}")
    logger.info(f"Trainable parameters: {count_trainable_parameters(model)}")

    logging.info(f"Start training")
    trainer = SpeechLLMTrainer(
        model=model,
        tokenizer=model.text_decoder.tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()


    if hasattr(model, 'config'):
        model.config.use_cache = True

    trainer.save_model(training_args.output_dir)


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="/net/tscratch/people/plgstefanop/speechllm/speechllm/config/test1.yaml")
    args = parser.parse_args()

    train(args.config)