"""Utilities for deriving RX/TX associations from channel tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class AssociationResult:
    """Container holding metadata for a RX/TX association decision.

    Attributes
    ----------
    rx_tx_matrix : np.ndarray
        Binary matrix of shape ``[num_ut, num_bs]`` where ``1`` indicates
        that the UE in row ``u`` is served by the BS in column ``b``.
    sector_ue_counts : np.ndarray
        Vector of length ``num_bs`` with the number of UEs assigned to each
        BS/sector.
    sector_available_mask : np.ndarray
        Boolean vector of length ``num_bs`` whose ``b``-th entry is ``True``
        when the BS ``b`` still has free streams (i.e., the number of
        associated UEs is strictly less than the available streams).
    """

    rx_tx_matrix: np.ndarray
    sector_ue_counts: np.ndarray
    sector_available_mask: np.ndarray


def _normalize_stream_limits(
    stream_limits: Union[int, Sequence[int]],
    num_bs: int,
) -> np.ndarray:
    """Broadcast the provided stream limits to a vector of length ``num_bs``."""

    if isinstance(stream_limits, (int, np.integer)):
        limits = np.full(num_bs, int(stream_limits), dtype=np.int32)
    else:
        limits = np.asarray(list(stream_limits), dtype=np.int32)
        if limits.ndim != 1:
            raise ValueError("stream_limits must be a scalar or 1-D sequence")
        if limits.size == 1:
            limits = np.full(num_bs, int(limits.item()), dtype=np.int32)
        elif limits.size != num_bs:
            raise ValueError(
                "stream_limits must provide one entry per BS or a broadcastable scalar"
            )

    if np.any(limits < 0):
        raise ValueError("stream limits must be non-negative")

    return limits


def build_rx_tx_association_from_h_slot(
    h_slot: Union[np.ndarray, tf.Tensor],
    stream_limits: Union[int, Sequence[int]],
) -> AssociationResult:
    """Derive a RX/TX association from a CFR tensor.

    The channel frequency response tensor ``h_slot`` is expected to follow the
    convention used across the PHY modules: ``[num_ut, num_ut_ant, num_bs,
    num_bs_ant, num_ofdm_symbols, num_subcarriers]``. The function averages the
    per-UE/per-BS power over all antenna and frequency dimensions, orders the BS
    candidates per UE, and greedily assigns each UE to the strongest BS that has
    free streams according to ``stream_limits``.

    Parameters
    ----------
    h_slot : np.ndarray | tf.Tensor
        Slot CFR with the convention described above.
    stream_limits : int | Sequence[int]
        Either a scalar specifying an identical limit for every BS or a
        sequence of length ``num_bs`` with per-BS stream limits.

    Returns
    -------
    AssociationResult
        dataclass containing the binary association matrix, the per-sector UE
        counts, and a mask that identifies BS that still have free streams.
    """

    h_tensor = tf.convert_to_tensor(h_slot)
    if h_tensor.shape.rank is None or h_tensor.shape.rank < 3:
        raise ValueError("h_slot must have rank >= 3 with UT and BS dimensions")

    num_ut = int(h_tensor.shape[0])
    num_bs = int(h_tensor.shape[2])

    if num_ut == 0 or num_bs == 0:
        empty_matrix = np.zeros((num_ut, num_bs), dtype=np.int32)
        counts = np.zeros(num_bs, dtype=np.int32)
        available = np.zeros(num_bs, dtype=bool)
        return AssociationResult(empty_matrix, counts, available)

    limits = _normalize_stream_limits(stream_limits, num_bs)

    power_axes = tuple(i for i in range(h_tensor.shape.rank) if i not in (0, 2))
    magnitude = tf.abs(h_tensor)
    power = tf.math.square(magnitude)
    if power_axes:
        power_ut_bs = tf.reduce_mean(power, axis=power_axes)
    else:
        power_ut_bs = power

    power_np = power_ut_bs.numpy()
    if power_np.shape != (num_ut, num_bs):
        power_np = np.reshape(power_np, (num_ut, num_bs))

    ranked_bs = np.argsort(-power_np, axis=1)
    rx_tx = np.zeros((num_ut, num_bs), dtype=np.int32)
    remaining_capacity = limits.astype(np.int32).copy()

    for ut_idx in range(num_ut):
        assigned = False
        for bs_idx in ranked_bs[ut_idx]:
            if remaining_capacity[bs_idx] > 0:
                rx_tx[ut_idx, int(bs_idx)] = 1
                remaining_capacity[bs_idx] -= 1
                assigned = True
                break
        if not assigned:
            raise RuntimeError(
                "Unable to associate UE within stream capacity constraints"
            )

    counts = rx_tx.sum(axis=0).astype(np.int32)
    available = counts < limits

    return AssociationResult(rx_tx, counts, available)


__all__ = [
    "AssociationResult",
    "build_rx_tx_association_from_h_slot",
]
