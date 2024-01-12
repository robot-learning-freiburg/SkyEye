import torch

from skyeye.utils.parallel import PackedSequence


def reorder_list(out, key):
    time_out = []
    for time_idx in range(len(out[key][0])):
        time_out_batch = []
        for batch_idx in range(len(out[key])):
            time_out_batch.append(out[key][batch_idx][time_idx])
        time_out.append(time_out_batch)
    return time_out


def iss_collate_fn(items):
    """Collate function for ISS batches"""
    out = {}
    if len(items) > 0:
        for key in items[0]:
            out[key] = [item[key] for item in items]
            if isinstance(items[0][key], torch.Tensor):
                out[key] = PackedSequence(out[key])
            # Handle the case where we have batches for multiple timesteps.
            # Basically reshape such lists so that we have a list of PackedSequences for each timestep.
            # This means that each PackedSequence contains only one timestep but for all elements in the batch
            elif isinstance(items[0][key], list) :
                reordered_list = reorder_list(out, key)
                # Not everything needs to be mapped into a PackedSequence
                if isinstance(items[0][key][0], torch.Tensor):
                    reordered_list = [PackedSequence(m) for m in reordered_list]
                out[key] = reordered_list

    return out
