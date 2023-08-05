__all__ = ["merge_multiple_timeseries_by_time", "truncate_timeseries_by_length"]

import torch

from startorch import constants as ct


def merge_multiple_timeseries_by_time(
    timeseries: list[dict[str, torch.Tensor]],
    time_key: str = ct.TIME,
) -> dict[str, torch.Tensor]:
    r"""Merges multiple time series by using the time.

    The sequence are merged based on the time information i.e. the
    merged time series is sorted by time. The input is a list of
    time series. Each time series is represented by a dictionary of
    ``torch.Tensor``s. All the values in the dictionary have to be
    of type ``torch.Tensor``. Each time series has to have a time
    tensor, and this time tensor is used to sort the merged time
    series. This function only supports the time tensor where
    the time is encoded as a single scalar (float, int, or long).
    The shape of the time tensor should be
    ``(batch_size, sequence_length, 1)`` or
    ``(batch_size, sequence_length)``. For example, the time in
    second can be used to encode the time information. All the
    tensors in each dictionary should have a shape
    ``(batch_size, sequence_length, *)`` where `*` means any
    number of dimensions. Only the sequence dimension can change
    between the time series. This function works for a variable
    number of features.

    Args:
    ----
        timeseries (list): Specifies the list of time series to merge.
            See the description above to know the format of the time
            series.
        time_key (str, optional): Specifies the key that contains the
            time information used to merge the time series.
            Default: ``'time'``

    Returns:
    -------
        dict: The merged time series. The dictionary has the same
            structure that the input time series.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch import constants as ct
        >>> from startorch.timeseries import merge_multiple_timeseries_by_time
        >>> merge_multiple_timeseries_by_time(
        ...     [
        ...         {
        ...             ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
        ...             ct.VALUE: torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
        ...         },
        ...         {
        ...             ct.TIME: torch.tensor([[[6], [12], [16], [24]]], dtype=torch.float),
        ...             ct.VALUE: torch.tensor([[21, 22, 23, 24]], dtype=torch.long),
        ...         },
        ...     ]
        ... )
        {'time': tensor([[[ 5.],
                 [ 6.],
                 [10.],
                 [12.],
                 [15.],
                 [16.],
                 [20.],
                 [24.],
                 [25.]]]), 'value': tensor([[11, 21, 12, 22, 13, 23, 14, 24, 15]])}
    """
    all_time_second = torch.cat([ts[time_key] for ts in timeseries], dim=1)
    sorted_values, indices = torch.sort(all_time_second, dim=1, stable=True)
    output = {time_key: sorted_values}

    # Merge the other keys.
    keys = set(timeseries[0].keys())
    keys.remove(time_key)
    for key in keys:
        inputs = torch.cat([ts[key] for ts in timeseries], dim=1)
        shapes = [1] * inputs.ndim
        shapes[:2] = inputs.shape[:2]
        output[key] = torch.gather(
            inputs, dim=1, index=indices.view(*shapes).repeat(1, 1, *inputs.shape[2:])
        )

    return output


def truncate_timeseries_by_length(
    timeseries: dict[str, torch.Tensor], max_seq_len: int
) -> dict[str, torch.Tensor]:
    r"""Truncates a time series by a maximum length.

    Each time series is represented by a dictionary of
    ``torch.Tensor``s. All the values in the dictionary have to be of
    type ``torch.Tensor``. All the tensors in each dictionary should
    have a shape ``(batch_size, sequence_length, *)`` where `*` means
    any number of dimensions. This function works for a variable
    number of features (keys).

    Args:
    ----
        timeseries (dict): Specifies the time series to truncate.
        max_seq_len (int): Specifies the maximum length of the
            time series.

    Returns:
    -------
        dict: The truncated time series where the sequence length
            is at most ``max_seq_len``.

    Raises:
    ------
        RuntimeError if ``max_seq_len`` is lowe than 1.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from startorch import constants as ct
        >>> from startorch.timeseries import truncate_timeseries_by_length
        >>> truncate_timeseries_by_length(
        ...     {
        ...         ct.TIME: torch.tensor([[[5], [10], [15], [20], [25]]], dtype=torch.float),
        ...         ct.VALUE: torch.tensor([[11, 12, 13, 14, 15]], dtype=torch.long),
        ...     },
        ...     max_seq_len=3,
        ... )
        {'time': tensor([[[ 5.],
                 [10.],
                 [15.]]]), 'value': tensor([[11, 12, 13]])}
    """
    if max_seq_len < 1:
        raise RuntimeError(
            f"`max_seq_len` has to be greater or equal to 1 (received: {max_seq_len:,})"
        )
    truncated_timeseries = {}
    for key, value in timeseries.items():
        truncated_timeseries[key] = value[:, :max_seq_len]
    return truncated_timeseries
