import torch

from syssim.compute.flop_counter import instruction_mix

aten = torch.ops.aten


def test_softmax_counts_transcendentals():
    x = torch.empty(32, 128, device="cpu")
    mma, fp32, transc = instruction_mix(aten._softmax, (x, -1, False), {}, x)
    assert mma == 0
    assert transc >= x.numel()        # >= 1 exp per element
    assert fp32 >= 0


def test_mm_is_pure_mma():
    a = torch.empty(64, 64, device="cpu")
    b = torch.empty(64, 64, device="cpu")
    out = torch.empty(64, 64, device="cpu")
    mma, fp32, transc = instruction_mix(aten.mm, (a, b), {}, out)
    assert mma == 2 * 64 ** 3
    assert transc == 0


def test_unregistered_op_is_zero_mix():
    a = torch.empty(8, device="cpu")
    assert instruction_mix(aten.view, (a, [8]), {}, a) == (0, 0, 0)
