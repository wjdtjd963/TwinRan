"""Utilities for building CFR tensors from ray-tracing assets."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import tensorflow as tf

try:  # pragma: no cover - optional dependency
    from channel_generator.hybrid_cluster.exceptions import HybridClusterError
except Exception:  # pylint: disable=broad-except
    class HybridClusterError(Exception):  # type: ignore[override]
        """Fallback when the channel generator package is unavailable."""


from .scenario_main import ScenarioState


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OFDMNumerology:
    """Container describing the OFDM grid expected by the simulator."""

    num_ut_ant: int
    num_bs_ant: int
    num_ofdm_symbols: int
    num_subcarriers: int
    subcarrier_spacing: float


class SupportsSetTopology(Protocol):
    """Minimal protocol satisfied by Sionna scenarios."""

    def set_topology(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        """Configure the network topology."""


class HybridGenerator(Protocol):
    """Duck-typed signature for :class:`HybridClusterGenerator`."""

    def __call__(
        self,
        *,
        user: Any,
        sample_times: tf.Tensor,
        normalize: bool,
        num_subcarriers: int,
        subcarrier_spacing: float,
    ) -> Tuple[tf.Tensor, tf.Tensor]:  # pragma: no cover - protocol
        """Return the frequency-domain channel and BS mask."""


def _ensure_vector(value: Optional[Iterable[float]]) -> np.ndarray:
    """Convert *value* to a flattened 3-D vector."""

    if value is None:
        return np.zeros(3, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        pad = np.zeros(3, dtype=np.float32)
        pad[: arr.size] = arr
        arr = pad
    return arr[:3].astype(np.float32)


def _stack_user_vectors(users: Sequence[Any], attr: str) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for user in users:
        mobility = getattr(user, "mobility", None)
        value = getattr(mobility, attr, None) if mobility is not None else None
        vectors.append(_ensure_vector(value))
    return np.stack(vectors, axis=0)


def _stack_bs_vectors(bs_list: Sequence[Any], attr: str) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for bs in bs_list:
        value = getattr(bs, attr, None)
        vectors.append(_ensure_vector(value))
    return np.stack(vectors, axis=0)


def _draw_random_channel(
    num_bs: int,
    numerology: OFDMNumerology,
    rng: Optional[tf.random.Generator],
) -> tf.Tensor:
    shape = (
        numerology.num_ut_ant,
        num_bs,
        numerology.num_bs_ant,
        numerology.num_ofdm_symbols,
        numerology.num_subcarriers,
    )
    if rng is not None:
        real = rng.normal(shape, dtype=tf.float32)
        imag = rng.normal(shape, dtype=tf.float32)
    else:
        real = tf.random.normal(shape, dtype=tf.float32)
        imag = tf.random.normal(shape, dtype=tf.float32)
    scale = tf.cast(1.0 / np.sqrt(2.0), tf.complex64)
    return tf.complex(real, imag) * scale


def _configure_topology(
    *,
    scenario: SupportsSetTopology,
    users: Sequence[Any],
    bs_list: Sequence[Any],
) -> None:
    ut_loc = _stack_user_vectors(users, "coordinate")[np.newaxis, ...]
    ut_orientations = _stack_user_vectors(users, "orientation")[np.newaxis, ...]
    ut_velocities = _stack_user_vectors(users, "velocity")[np.newaxis, ...]
    bs_loc = _stack_bs_vectors(bs_list, "coordinate")[np.newaxis, ...]
    bs_orientations = _stack_bs_vectors(bs_list, "orientation")[np.newaxis, ...]
    in_state = tf.zeros([1, len(users)], dtype=tf.bool)
    scenario.set_topology(
        ut_loc=tf.convert_to_tensor(ut_loc, dtype=tf.float32),
        bs_loc=tf.convert_to_tensor(bs_loc, dtype=tf.float32),
        ut_orientations=tf.convert_to_tensor(ut_orientations, dtype=tf.float32),
        bs_orientations=tf.convert_to_tensor(bs_orientations, dtype=tf.float32),
        ut_velocities=tf.convert_to_tensor(ut_velocities, dtype=tf.float32),
        in_state=in_state,
    )


def get_cfr_from_ray_tracing(
    scenario_state: ScenarioState,
    numerology: OFDMNumerology,
    scenario: SupportsSetTopology,
    hybrid_generator: HybridGenerator,
    *,
    normalize: bool = False,
    rng: Optional[tf.random.Generator] = None,
) -> Tuple[Optional[tf.Tensor], List[Dict[str, Any]]]:
    """Build the slot-level CFR tensor using ray-tracing PKL files.

    Parameters
    ----------
    scenario_state:
        Live :class:`ScenarioState` snapshot describing users and BSes.
    numerology:
        OFDM parameters required by the simulator pipeline.
    scenario:
        Configured Sionna system-level scenario implementing ``set_topology``.
    hybrid_generator:
        Instance of :class:`HybridClusterGenerator` (or a compatible object).
    normalize:
        Forwarded to :meth:`HybridClusterGenerator.__call__`.
    rng:
        Optional :class:`tf.random.Generator` used for deterministic fallback
        channels.

    Returns
    -------
    tuple
        ``(h_slot, metadata_list)`` where ``h_slot`` matches the simulator
        convention ``[num_ut, num_ut_ant, num_bs, num_bs_ant, num_ofdm_symbols,
        num_subcarriers]``. ``metadata_list`` keeps track of the ray-tracing
        source used for every UE.
    """

    if scenario_state is None:
        raise ValueError("scenario_state must be provided")

    users = list(getattr(scenario_state, "users", []) or [])
    bs_list = list(getattr(scenario_state, "bs_list", []) or [])

    if len(users) == 0 or len(bs_list) == 0:
        LOGGER.warning("No users or BS instances available for CFR generation")
        return None, []

    _configure_topology(scenario=scenario, users=users, bs_list=bs_list)

    sample_times = tf.range(numerology.num_ofdm_symbols, dtype=tf.float32)
    num_bs = len(bs_list)
    metadata: List[Dict[str, Any]] = []
    channels: List[tf.Tensor] = []

    for idx, user in enumerate(users):
        pkl_path = getattr(user, "pkl", None)
        entry: Dict[str, Any] = {
            "ue_index": idx,
            "ue_id": getattr(user, "user_id", None),
            "rt_file": pkl_path,
            "from_ray_tracing": False,
        }
        use_rt = pkl_path is not None and os.path.isfile(pkl_path)

        if use_rt:
            try:
                h_freq, _ = hybrid_generator(
                    user=user,
                    sample_times=sample_times,
                    normalize=normalize,
                    num_subcarriers=numerology.num_subcarriers,
                    subcarrier_spacing=numerology.subcarrier_spacing,
                )
                user_cfr = tf.squeeze(h_freq, axis=(0, 1))
                expected_shape = (
                    numerology.num_ut_ant,
                    num_bs,
                    numerology.num_bs_ant,
                    numerology.num_ofdm_symbols,
                    numerology.num_subcarriers,
                )
                user_cfr = tf.ensure_shape(user_cfr, expected_shape)
                channels.append(user_cfr)
                entry["from_ray_tracing"] = True
            except (HybridClusterError, OSError, ValueError) as exc:
                LOGGER.warning(
                    "Falling back to random channel for UE %s due to %s",
                    entry["ue_id"] or idx,
                    exc,
                )
                entry["error"] = str(exc)
                channels.append(_draw_random_channel(num_bs, numerology, rng))
        else:
            if pkl_path is not None:
                entry["error"] = "Ray-tracing file missing"
            channels.append(_draw_random_channel(num_bs, numerology, rng))

        metadata.append(entry)

    if not channels:
        return None, metadata

    h_slot = tf.stack(channels, axis=0)
    return h_slot, metadata


__all__ = ["OFDMNumerology", "get_cfr_from_ray_tracing"]
