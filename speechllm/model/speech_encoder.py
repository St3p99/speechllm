import random
from typing import Optional, Tuple, Union

from speechllm.model.downsample import Downsample
import torch
import torch.nn as nn
from transformers import WavLMConfig, WavLMModel


class WavLMEncoder(nn.Module):
    """
    Wrapper class for the WavLM Large model from Hugging Face Transformers.

    Parameters
    ----------
    _name_or_path : str, optional
        Path or identifier for the pretrained WavLM model from Hugging Face.
        Defaults to "microsoft/wavlm-large".
    stack_size : int, optional
        Size of the stack for downsampling. Defaults to 5.
    """

    def __init__(
        self, _name_or_path: str = "microsoft/wavlm-large", stack_size: int = 5
    ) -> None:
        """
        Initialize WavLMEncoder with the specified or default model path.
        """
        super().__init__()
        if not isinstance(_name_or_path, str):
            raise TypeError("Expected '_name_or_path' to be a string.")
        if not isinstance(stack_size, int) or stack_size <= 0:
            raise ValueError("Expected 'stack_size' to be a positive integer.")

        self.config = WavLMConfig.from_pretrained(_name_or_path)
        self.model = WavLMModel.from_pretrained(_name_or_path)
        self.downsample = Downsample(stack_size=stack_size)

    @property
    def input_sampling_rate(self) -> int:
        """Return the expected input sampling rate."""
        return 16000  # FIXME: Hardcoded

    @property
    def output_sampling_rate(self) -> float:
        """Calculate the output sampling rate after convolution layers."""
        stride = self.config.conv_stride  # list of stride values for each conv layer
        total_stride = torch.prod(torch.tensor(stride)).item()
        return self.input_sampling_rate / total_stride

    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the model."""
        return self.config.hidden_size

    @property
    def output_hidden_size(self) -> int:
        """Return the actual output hidden size after downsampling."""
        return self.config.hidden_size * self.downsample.stack_size

    @torch.no_grad()
    def forward(
        self,
        waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(waveforms, torch.Tensor):
            raise TypeError("Expected 'waveforms' to be a torch.Tensor.")

        if waveforms.dim() != 2:
            raise ValueError(
                "Expected 'waveforms' to have shape (batch_size, samples)."
            )

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(waveforms, dtype=torch.bool)

        # Convert attention mask to the format expected by WavLM (float, 0 for padding)
        wavlm_attention_mask = attention_mask.float()

        model_output = self.model(
            waveforms, attention_mask=wavlm_attention_mask, return_dict=True
        )
        x = model_output.last_hidden_state  # shape: (B, T, D)

        # Calculate downsampling factor for attention mask
        stride = self.config.conv_stride
        total_stride = torch.prod(torch.tensor(stride)).item()

        # Create attention mask for the encoded features
        batch_size = attention_mask.shape[0]
        encoded_seq_len = x.shape[1]

        # Create attention mask for encoded features
        encoded_attention_mask = torch.zeros(
            batch_size, encoded_seq_len, dtype=torch.bool, device=x.device
        )

        for i in range(encoded_seq_len):
            start_idx = i * total_stride
            end_idx = min(start_idx + total_stride, attention_mask.shape[1])
            # If any part of the window is valid, mark the encoded position as valid
            encoded_attention_mask[:, i] = attention_mask[:, start_idx:end_idx].any(
                dim=1
            )

        # Apply the mask to the encoded features
        x = x * encoded_attention_mask.unsqueeze(-1).float()

        # Apply stack-based downsampling
        x = self.downsample(x)  # shape: (B, T/stack_size, D*stack_size)

        # Create attention mask for the final downsampled features
        final_seq_len = x.shape[1]
        final_attention_mask = torch.zeros(
            batch_size, final_seq_len, dtype=torch.bool, device=x.device
        )

        for i in range(final_seq_len):
            start_idx = i * self.downsample.stack_size
            end_idx = min(
                start_idx + self.downsample.stack_size, encoded_attention_mask.shape[1]
            )
            # If any part of the stacked window is valid, mark the final position as valid
            final_attention_mask[:, i] = encoded_attention_mask[
                :, start_idx:end_idx
            ].any(dim=1)

        # Apply final mask to the output features
        x = x * final_attention_mask.unsqueeze(-1).float()

        if return_attention_mask:
            return x, final_attention_mask
        return x


def create_test_batch(
    batch_size: int,
    sampling_rate: int,
    min_duration: float = 0.5,
    max_duration: float = 4.0,
):
    """
    Create a batch of random waveforms with different lengths for testing.

    Parameters
    ----------
    batch_size : int
        Number of audio samples in the batch.
    sampling_rate : int
        Sampling rate of the audio.
    min_duration : float
        Minimum duration in seconds.
    max_duration : float
        Maximum duration in seconds.

    Returns
    -------
    tuple
        (audios_tensor, attention_masks_tensor) where both are torch.Tensor
    """
    audios = []
    for i in range(batch_size):
        audio_length_in_sec = random.uniform(min_duration, max_duration)
        audio_length_in_samples = int(audio_length_in_sec * sampling_rate)
        audio = torch.randn(audio_length_in_samples)
        audios.append(audio)

    # Pad the audios to the same length
    max_length = max(len(audio) for audio in audios)
    min_length = min(len(audio) for audio in audios)

    print(f"Audio lengths - Max: {max_length}, Min: {min_length}")

    audio_attention_masks = []
    for i in range(batch_size):
        audio_len = len(audios[i])
        if audio_len < max_length:
            # Pad audio to max_length
            audios[i] = torch.cat(
                [
                    audios[i],
                    torch.zeros(max_length - audio_len, device=audios[i].device),
                ]
            )
        # Create attention mask: True for valid audio, False for padding
        mask = torch.ones(max_length, dtype=torch.bool, device=audios[i].device)
        mask[audio_len:] = False
        audio_attention_masks.append(mask)

    # Convert to tensor stacks
    audios_tensor = torch.stack(audios)
    audio_attention_masks_tensor = torch.stack(audio_attention_masks)

    return audios_tensor, audio_attention_masks_tensor


def test_encoders():
    """
    Test WavLM encoder with sample data.
    """
    print("=" * 50)
    print("Testing Speech Encoders")
    print("=" * 50)

    batch_size = 2

    # Test WavLM encoder
    print("\n1. Testing WavLM Encoder:")
    print("-" * 30)
    try:
        speech_encoder = WavLMEncoder(stack_size=5)
        audios_tensor, audio_attention_masks_tensor = create_test_batch(
            batch_size, speech_encoder.input_sampling_rate
        )

        print(f"Input shape: {audios_tensor.shape}")
        print(f"Input sampling rate: {speech_encoder.input_sampling_rate}")
        print(f"Output sampling rate: {speech_encoder.output_sampling_rate:.2f}")
        print(f"Hidden size: {speech_encoder.hidden_size}")
        print(f"Output hidden size: {speech_encoder.output_hidden_size}")

        speech_encoder_output, speech_encoder_attention_mask = speech_encoder(
            audios_tensor, audio_attention_masks_tensor, return_attention_mask=True
        )

        print(f"WavLM output shape: {speech_encoder_output.shape}")
        print(f"WavLM attention mask shape: {speech_encoder_attention_mask.shape}")
        print(f"WavLM attention mask sum: {speech_encoder_attention_mask.sum(dim=1)}")
        print(f"WavLM attention mask: {speech_encoder_attention_mask}")

    except Exception as e:
        print(f"WavLM Encoder test failed: {e}")

    print("\n" + "=" * 50)
    print("Testing completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_encoders()
