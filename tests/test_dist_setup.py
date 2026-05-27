import torch.distributed as dist


def test_fake_pg_sets_world_size():
    from syssim.training.dist_setup import init_fake_process_group, destroy_process_group

    init_fake_process_group(world_size=32, rank=5)
    try:
        assert dist.is_initialized()
        assert dist.get_world_size() == 32
        assert dist.get_rank() == 5
    finally:
        destroy_process_group()
    assert not dist.is_initialized()


def test_fake_pg_supports_new_group():
    from syssim.training.dist_setup import init_fake_process_group, destroy_process_group

    init_fake_process_group(world_size=8, rank=0)
    try:
        g = dist.new_group(ranks=[0, 1, 2, 3])
        assert g is not None
    finally:
        destroy_process_group()
