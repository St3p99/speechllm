# SpeechLLM

A multimodal speech-to-text framework that implements a SpeechLLM architecture for automatic speech recognition, combining speech encoders with large language models through learnable projection layers.

## Overview

SpeechLLM implements a neural architecture for automatic speech recognition by integrating four key components: speech encoder → downsampling → projector → LLM. This enables end-to-end training for speech recognition while leveraging pre-trained language models.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.3.1+
- CUDA (recommended for GPU training)

### Installation

```bash
git clone <repository-url>
cd speechllm

pip install --upgrade pip
pip install -e .

# For training capabilities, install with training dependencies
pip install -e ".[train]"
```

## Quick Start

```python
from speechllm import SpeechLLM, SpeechLLMConfig

# Create and initialize model
config = SpeechLLMConfig(
    speech_encoder_name_or_path="microsoft/wavlm-large",
    text_decoder_name_or_path="meta-llama/Llama-3.2-1B",
    downsample_factor=5
)
model = SpeechLLM(config=config)
```

See [Training](#training) section for training instructions.

## Configuration

### Training Configuration (`config/train.yaml`)

The training is configured through a YAML file with three main sections:

#### Data Configuration
```yaml
data:
  data_path: "parler-tts/libritts_r_filtered"  # HuggingFace dataset path or local path
  subset: "clean"           # Dataset subset (if applicable)
  split: "train.clean.100"  # Dataset split to use
  amount: ":100%"           # Percentage of dataset to use (e.g., ":10%", ":1000")
  remap_keys: {"text_normalized": "transcription"}  # Map dataset fields to expected names
  sampling_rate: 16000      # Target audio sampling rate
  min_duration: 0.01        # Minimum audio duration (seconds)
  max_duration: 15.0        # Maximum audio duration (seconds)
  num_proc_for_preprocessing: 4  # Parallel preprocessing workers
```

**Notes:**
- Use HuggingFace dataset names (e.g., `"parler-tts/libritts_r_filtered"`) or local paths
- `remap_keys` maps dataset fields to expected names (`"text_normalized": "transcription"`)

#### Training Parameters
```yaml
training:
  output_dir: ./output/
  freeze_modules: ["encoder", "text_decoder"]  # Modules to freeze during training
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 1
  learning_rate: 1e-4
  warmup_steps: 1000
  bf16: true  # Use bfloat16 precision
  lr_scheduler_type: cosine
  optim: adamw_torch
  max_grad_norm: 1.0
```

#### Model Architecture
```yaml
model:
  speech_encoder_name_or_path: "microsoft/wavlm-large"
  text_decoder_name_or_path: "meta-llama/Llama-3.2-1B"
  downsample_factor: 5  # Speech feature downsampling
  projector_n_layers: 2  # Projector depth
  projector_activation: "relu"
  projector_hidden_size: 2048
```



## Architecture

```
Audio Input → Speech Encoder → Downsampling → Projector → LLM → Text Output
                                                        ↑
                                                    Text Prompt
```

The four-component pipeline processes audio through encoders and projectors before the LLM generates transcriptions. See [Prompt Format](#prompt-format) for details on how components interact during processing.

## Dataset Format

The framework expects datasets with the following structure:

```python
{
    "audio": {
        "array": [audio_array],           # Raw audio data as numpy array or list
        "sampling_rate": 16000            # Original sampling rate
    },
    "transcription": "Hello world",      # Ground truth transcription text
    "duration": 2.5,                     # Audio duration in seconds (optional)
    # ... other metadata fields
}
```

Use remap_keys to map fields in your dataset to the expected names. For example, if your dataset uses "text" instead of "transcription", set `remap_keys: {"text": "transcription"}` in your config to ensure the framework reads the correct field.

## Prompt Format

SpeechLLM uses a specific conversation template for speech recognition tasks. The system processes audio input with a structured prompt format:

### Input Prompt Structure
```
<audio>
transcribe input speech to English text: 
```

### Complete Conversation Format
```python
[
    {
        "from": "human", 
        "value": "<audio>\ntranscribe input speech to English text: "
    },
    {
        "from": "assistant", 
        "value": "Hello world"  # Ground truth transcription
    }
]
```

**Key Points:**
- **`<audio>`**: Placeholder token replaced by projected speech features during forward pass
- **Textual prompt**: Task instruction for the LLM (`"transcribe input speech to English text: "`)
- **Processing**: Audio → Speech pipeline → Replace `<audio>` token → LLM generates transcription

## Training

```bash
# Basic training
python speechllm/train.py --config config/train.yaml

# Multi-GPU training  
torchrun --nproc_per_node=<n_gpus> speechllm/train.py --config config/train.yaml
```

**Custom Dataset Setup:**
1. Format dataset with `"audio"` and `"transcription"` fields (or use `remap_keys`)
2. Update `data_path` in `config/train.yaml` 
3. Run training command above

## Contact

- Author: Stefano Perna
- Email: pernastefano99@gmail.com
