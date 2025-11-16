import importlib.util
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sionna", PROJECT_ROOT / "__init__.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # type: ignore[arg-type]
sys.modules.setdefault("sionna", module)

from sionna.sys.association import build_rx_tx_association_from_h_slot


def _slot_from_power(power_matrix: np.ndarray) -> np.ndarray:
    """Utility that embeds a power matrix into an h_slot tensor."""

    amplitudes = np.sqrt(np.asarray(power_matrix, dtype=np.float32))
    h_slot = amplitudes.astype(np.complex64)
    return h_slot[:, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]


def test_association_prefers_best_bs_with_capacity():
    power = np.array(
        [
            [10.0, 5.0],
            [1.0, 9.0],
            [2.0, 8.0],
        ],
        dtype=np.float32,
    )
    result = build_rx_tx_association_from_h_slot(_slot_from_power(power), [2, 1])

    expected = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(result.rx_tx_matrix, expected)
    np.testing.assert_array_equal(result.sector_ue_counts, np.array([2, 1], dtype=np.int32))


def test_counts_and_capacity_mask_report_remaining_streams():
    power = np.array(
        [
            [5.0, 3.0],
            [4.0, 1.0],
        ],
        dtype=np.float32,
    )
    result = build_rx_tx_association_from_h_slot(_slot_from_power(power), 3)

    np.testing.assert_array_equal(result.sector_ue_counts, np.array([2, 0], dtype=np.int32))
    np.testing.assert_array_equal(result.sector_available_mask, np.array([True, True]))


def test_association_raises_when_capacity_insufficient():
    power = np.array(
        [
            [3.0, 2.0],
            [2.5, 1.0],
            [1.5, 0.5],
        ],
        dtype=np.float32,
    )

    with pytest.raises(RuntimeError):
        build_rx_tx_association_from_h_slot(_slot_from_power(power), [1, 1])
