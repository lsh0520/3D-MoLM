import torch
from typing import Any, Iterable, Iterator, List, Optional, Sized, Tuple, Union, Dict
from torch import Tensor
import torch.nn.functional as F
from lavis.common.dist_utils import is_dist_avail_and_initialized
from model.help_funcs import pad_and_concat
from pytorch_lightning import strategies
from lightning_fabric.utilities.types import _PATH
from deepspeed.runtime.data_pipeline.data_routing.helper import remove_random_ltd_state_dict


'''
overwrite the function in deepspeed
'''


### start overwrite ###
def module_state_dict(self, destination=None, prefix="", keep_vars=False, exclude_frozen_parameters=False):
    sd = self.module.state_dict(destination, prefix, keep_vars)
    # Remove frozen parameter weights from state_dict if specified
    if exclude_frozen_parameters:
        to_be_removed = []
        for n in sd:
            try: 
                if not self.module.get_parameter(n).requires_grad:
                    to_be_removed.append(n)
            except AttributeError:
                to_be_removed.append(n)
        for key in to_be_removed:
            sd.pop(key)
    if self.random_ltd_enabled():
        sd = remove_random_ltd_state_dict(sd)
    return sd
from deepspeed import DeepSpeedEngine
DeepSpeedEngine.module_state_dict = module_state_dict
### end overwrite ###


class MyDeepSpeedStrategy(strategies.DeepSpeedStrategy):
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ):
        ## use this method when no need to save the optimizer states 
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
            storage_options: parameter for how to save to st
            orage, passed to ``CheckpointIO`` plugin
        """
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    def save_checkpoint_v2(self, checkpoint: Dict, filepath: _PATH, storage_options: Optional[Any] = None) -> None:
        ## use this method when need to save the optimizer states 
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: The checkpoint state dictionary
            filepath: write-target file's path
            storage_options: not used for ``DeepSpeedStrategy`` as ``CheckpointIO`` is not used

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in
        """
        # broadcast the filepath from rank 0 to ensure all the states are saved in a common filepath
        filepath = self.broadcast(filepath)
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}` as `CheckpointIO` is not used."
            )

        if self.zero_stage_3 and self._multi_device and self.is_global_zero:
            print(
                "Warning: When saving the DeepSpeed Stage 3 checkpoint, "
                "each worker will save a shard of the checkpoint within a directory. "
                "If a single file is required after training, "
                "see https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#"
                "deepspeed-zero-stage-3-single-file for instructions."
            )
        # Use deepspeed's internal checkpointing function to handle partitioned weights across processes
        # dump states as a checkpoint dictionary object
        _exclude_keys = ["state_dict", "optimizer_states"]
        checkpoint = {k: v for k, v in checkpoint.items() if k not in _exclude_keys}
        self.deepspeed_engine.save_checkpoint(filepath, client_state=checkpoint, tag="checkpoint", exclude_frozen_parameters=True)


@torch.no_grad()
def pl_concat_all_gather(tensor, padding=False, fill_value=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = gather_all_tensors(tensor)
    if padding:
        output = pad_and_concat(tensors_gather, fill_value=fill_value).detach()
    else:
        output = torch.cat(tensors_gather, dim=0)
    return output


def gather_all_tensors(*args: Any, **kwargs: Any) -> Any:
    return _gather_all_tensors(*args, **kwargs)


def _gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Function to gather all tensors from several DDP processes onto a list that is broadcasted to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: The value to sync
        group: The process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: List with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    if group is None:
        group = torch.distributed.group.WORLD

    # Convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)

    # If the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result


def _simple_gather_all_tensors(result: Tensor, group: Any, world_size: int) -> List[Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result