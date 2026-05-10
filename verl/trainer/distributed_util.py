from verl.utils.device import is_npu_available

def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    # NOTE: If it is necessary to support weight synchronization with the sglang backend in the future,
    # the following can be used:
    # from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    # from sglang.srt.distributed.utils import statelessprocessgroup
    if is_npu_available:
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    pynccl = PyHcclCommunicator(pg, device=device)
    return pynccl
