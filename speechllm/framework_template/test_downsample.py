"""
Test downsample.py.
"""

import torch
from speechllm import Downsample


def test_downsampling_basic():
    # Test with batch_size=1, length=5, feature_num=2, stack_size=3
    batch_size = 1
    length = 5
    feature_num = 2
    stack_size = 3
    vectors = torch.arange(
        batch_size * length * feature_num, dtype=torch.float32
    ).reshape(batch_size, length, feature_num)
    downsample = Downsample(stack_size)
    output = downsample.forward(vectors)
    # Should pad to length 6, then reshape to (1, 2, 6)
    expected = torch.tensor(
        [
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 0, 0],
            ]
        ],
        dtype=torch.float32,
    ).reshape(batch_size, 2, 6)
    assert output.shape == (batch_size, 2, stack_size * feature_num)
    # Check values
    assert torch.allclose(output, expected)


def test_downsampling_no_padding():
    # Test with no padding needed
    batch_size = 2
    length = 4
    feature_num = 1
    stack_size = 2
    vectors = torch.arange(
        batch_size * length * feature_num, dtype=torch.float32
    ).reshape(batch_size, length, feature_num)
    downsample = Downsample(stack_size)
    output = downsample.forward(vectors)
    # Should reshape to (2, 2, 2)
    expected = torch.tensor(
        [
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
        ],
        dtype=torch.float32,
    )
    assert output.shape == (batch_size, 2, stack_size * feature_num)
    assert torch.allclose(output, expected)
