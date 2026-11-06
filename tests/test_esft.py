from torch import nn

from esft import to_buffer, to_param


def test_to_buffer_and_to_param_roundtrip():
    linear = nn.Linear(4, 2)
    assert len(list(linear.parameters())) == 2
    assert len(list(linear.buffers())) == 0

    to_buffer(linear)
    assert len(list(linear.parameters())) == 0
    assert len(list(linear.buffers())) == 2
    assert hasattr(linear, 'param_list')
    assert set(linear.param_list) == {'weight', 'bias'}

    # Calling to_buffer again should not duplicate names in param_list
    to_buffer(linear)
    assert len(linear.param_list) == 2

    to_param(linear)
    assert len(list(linear.parameters())) == 2
    assert len(list(linear.buffers())) == 0
