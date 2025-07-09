"""
Framework for the solution to the coding task from Samsung's AI Center in
Cambridge.

It's meant to be useful, but feel free to use this or to ignore this.

Package requirements:
torch datasets transformers librosa soundfile
"""

import torch
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import datasets

# Select device: try CUDA, then MPS (Apple Silicon), then CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Feel free to change the dataset.
def load_librispeech():
    # Load LibriSpeech train-clean-100 dataset
    librispeech = datasets.load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100",
        # Load streaming so that the whole of Librispeech is not downloaded.
        # If you have Librispeech cached, you can remove this.
        streaming=True,
    )
    assert isinstance(librispeech, datasets.IterableDataset)  # Better type hints.
    return librispeech


# Load processor and model from Huggingface.
# Feel free to change the model.
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
assert isinstance(processor, Wav2Vec2Processor)  # Better type hints.
speech_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Load a small LLM from Huggingface (distilgpt2).
# Feel free to change the model.
llm = AutoModelForCausalLM.from_pretrained("distilgpt2")
llm_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")


# TODO Put your model together.
# TODO Other things.


# The main training loop.
# Loop only a few times.
# for index, ... in zip(range(5), ...):
#     ...
