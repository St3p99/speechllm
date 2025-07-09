import logging
import torch
import torchaudio.transforms as T
from datasets import Dataset, load_dataset, concatenate_datasets, DownloadMode
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Union
import os

from speechllm.arguments import DataArguments
from speechllm.dataset.utils import preprocess
from speechllm.constants import DEFAULT_AUDIO_TOKEN

logger = logging.getLogger(__name__)


class SpeechDataset(TorchDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        self.tokenizer = tokenizer

        # Unpack all DataArguments fields as attributes of this class
        self.data_path = data_args.data_path
        self.split = data_args.split
        self.amount = data_args.amount
        self.min_duration = data_args.min_duration
        self.max_duration = data_args.max_duration
        self.remap_keys = data_args.remap_keys if data_args.remap_keys is not None else {}
        self.sampling_rate = data_args.sampling_rate
        self.num_proc_for_preprocessing = data_args.num_proc_for_preprocessing

        self.dataset = self._load_dataset()
        # Apply preprocessing
        self.dataset = self._preprocess_dataset()

        logger.info(f"Loaded {len(self.dataset)} samples")

    def _load_dataset(self):
        if self.data_path == None:
            # For demo purposes, hardcoding parler-tts/libritts_r_filtered clean train.clean.100
            dataset = load_dataset(
                "parler-tts/libritts_r_filtered",
                "clean",
                split="train.clean.100",
            )
            return dataset

        # load the dataset from local parquet files
        data_path = self._get_dataset_path()

        # Parse amount parameter
        if self.amount is None:
            split_str = self.split
        elif isinstance(self.amount, str):
            # Handle :10, :100, etc.
            split_str = f"{self.split}[{self.amount}]"
        else:
            # Handle other formats
            split_str = self.split

        datasets = []
        if isinstance(data_path, list):
            for path in data_path:
                ds = load_dataset(
                    "parquet",
                    data_files={self.split: path},
                    split=split_str,
                )
                datasets.append(ds)
            dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
        else:
            dataset = load_dataset(
                "parquet",
                data_files={self.split: data_path},
                split=split_str,
            )
        return dataset

    def _get_dataset_path(self):
        """Get the path to the dataset parquet file."""
        data_path = Path(self.data_path)
        if self.split == "train":
            filename = "train.clean.100.parquet"
        elif self.split == "train_full":
            filename = ["train.clean.100.parquet", "train.clean.360.parquet", "train.other.500.parquet"]
        elif self.split == "validation":
            filename = "dev.clean.parquet"
        elif self.split == "test":
            filename = "test.clean.parquet"
        else:
            filename = f"{self.split}/data.parquet"
        return str(data_path / filename) if isinstance(filename, str) else [str(data_path / f) for f in filename]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.dataset[i]

        # Process text input to create input_ids and labels
        data_dict = self._process_text_input(data)

        # Process audio if present
        if "audio" in data and data["audio"] is not None:
            # Handle different audio formats
            audio_input, sr_input = self._process_audio_input(data["audio"])

            data_dict.update({
                "audio": [audio_input],
                "audio_sr": [sr_input]
            })

        return data_dict

    def _preprocess_dataset(self):
        """Apply filtering and preprocessing to the dataset."""
        if self.min_duration is not None or self.max_duration is not None:
            self.dataset = self.dataset.filter(self._filter_by_duration, num_proc=self.num_proc_for_preprocessing)

        self.dataset = self.dataset.map(
            lambda example: self._remap_keys_in_example(example),
            num_proc=self.num_proc_for_preprocessing
        )
        self.dataset = self.dataset.map(
            self._prepare_asr_example,
            num_proc=self.num_proc_for_preprocessing
        )
        return self.dataset

    def _prepare_asr_example(self, example):
        """Prepare ASR example."""
        example["conversations"] = self._create_asr_conversation(example["transcription"])
        return example

    def _filter_by_duration(self, example):
        """Filter examples by audio duration."""
        # Try to get duration from different possible keys
        duration = None
        if 'duration' in example:
            duration = example['duration']
        elif 'audio' in example and example['audio'] is not None:
            if isinstance(example['audio'], dict) and 'array' in example['audio']:
                duration = len(example['audio']['array']) / example['audio']['sampling_rate']
        else:
            raise ValueError(f"No audio input found in example: {example}")

        if duration is None:
            return True  # Keep if we can't determine duration

        if self.min_duration is not None and duration < self.min_duration:
            return False
        if self.max_duration is not None and duration > self.max_duration:
            return False
        return True

    def _remap_keys_in_example(self, example):
        """Remap keys in the example based on remap_keys mapping."""
        for old_key, new_key in self.remap_keys.items():
            if old_key in example:
                example[new_key] = example.pop(old_key)
        return example

    def _resample_audio(self, audio: torch.Tensor, sr: int, target_sr: int = None):
        """Resample audio to target sampling rate."""
        if target_sr is None:
            target_sr = self.sampling_rate

        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            audio = resampler(audio)

        return audio, target_sr

    def _process_audio_input(self, audio_data):
        """Process audio input with resampling."""
        if isinstance(audio_data, dict) and 'array' in audio_data:
            # Handle HuggingFace datasets audio format
            audio_array = torch.as_tensor(audio_data["array"]).to(torch.float32)
            sr = audio_data["sampling_rate"]
        elif isinstance(audio_data, torch.Tensor):
            # Handle tensor input
            audio_array = audio_data.to(torch.float32)
            sr = self.sampling_rate  # Assume target sampling rate
        else:
            raise ValueError(f"Unsupported audio format: {type(audio_data)}")

        # Resample if needed
        target_sr = getattr(self, 'audio_input_sampling_rate', self.sampling_rate)
        audio, sr = self._resample_audio(audio_array, sr, target_sr)

        return audio, sr

    def _process_text_input(self, data):
        """Process text input and create input_ids and labels."""
        data_dict = {}

        # Handle different conversation formats
        if "conversations" in data:
            sources = [data["conversations"]]
        elif "transcription" in data:
            # Create ASR conversation format
            sources = [self._create_asr_conversation(data["transcription"])]
        else:
            raise ValueError("No valid text input found in data")

        # Tokenize using the preprocess function
        tokenized = preprocess(
            sources,
            self.tokenizer,
            getattr(self, 'conversation_version', 'llama_3_1'),
            has_audio=data.get("audio", None) is not None
        )

        data_dict.update({
            "input_ids": tokenized["input_ids"][0],
            "labels": tokenized["labels"][0],
        })

        return data_dict

    def _create_asr_conversation(self, transcription: str):
        """Create ASR conversation format from transcription."""
        return [
            {
                "from": "human",
                "value": f"{DEFAULT_AUDIO_TOKEN}\ntranscribe input speech to English text: "
            },
            {
                "from": "assistant",
                "value": transcription
            }
        ]


from transformers import AutoTokenizer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    pad_token, pad_token_id = "<|finetune_right_pad_id|>", 128004
    tokenizer.pad_token = pad_token
    tokenizer.pad_token_id = pad_token_id
    data_args = DataArguments()
    dataset = SpeechDataset(
        tokenizer=tokenizer,
        data_args=data_args,
    )