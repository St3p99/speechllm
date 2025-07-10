"""
Torch module that downsamples input, i.e. it stacks every n vectors.
"""

import torch


# From the paper (Sec. 3.1) https://arxiv.org/pdf/2110.13900:
# We stack every n consecutive frames to form a new feature vector,
# reducing the sequence length by a factor of n while increasing the hidden size accordingly.
class Downsample(torch.nn.Module):
    """
    Torch module that downsamples input, i.e. it stacks every n vectors, and
    therefore reduces the length of the sequence by a factor of (almost) n.
    If the length does not divide by n, extra frames that are all zero are
    appended.

    :param stack_size:
        The number of adjacent frames to stack.
    """

    def __init__(self, stack_size: int):
        super().__init__()
        self.stack_size = stack_size

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        batch_size, length, feature_num = vectors.shape
        stack_size = self.stack_size
        # Calculate the new length after stacking. Round up.
        new_length = (length + stack_size - 1) // stack_size

        # Pad vectors with zeros if necessary along the length dimension.
        pad_length = new_length * stack_size - length
        if pad_length > 0:
            vectors = torch.nn.functional.pad(vectors, (0, 0, 0, pad_length))

        vectors = vectors.view(batch_size, new_length, stack_size, feature_num)
        # Merge the last two dimensions into a single dimension.
        vectors = vectors.reshape(batch_size, new_length, stack_size * feature_num)
        return vectors
