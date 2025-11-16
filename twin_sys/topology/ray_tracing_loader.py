"""Utilities for loading ray-tracing datasets and initializing hybrid generators."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from channel_generator.HybridClusterGenerator import HybridClusterGenerator
from sionna.phy.channel.tr38901 import (
    LSPGenerator,
    PanelArray,
    UMiScenario,
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_RT_FILE = (
    Path(__file__).resolve().parents[2]
    / "channel_generator"
    / "hybrid_cluster"
    / "rt_outputs.pkl"
)


@dataclass(slots=True)
class RayTracingLoaderResult:
    """Container storing artifacts produced by :func:`load_rt_data_and_init_generator`.

    Attributes
    ----------
    available:
        True when the ray-tracing dataset could be loaded successfully.
    rt_params_np:
        Dictionary with numpy tensors shaped with batch dimensions.
    los_flag_np:
        Boolean mask of shape ``[B, UT, BS]``.
    ue_coordinates:
        Array of UE coordinates shaped ``[num_ut, 3]``.
    bs_coordinates:
        Array of BS coordinates shaped ``[num_bs, 3]``.
    ut_panel_array / bs_panel_array:
        Panel array descriptions used to instantiate the scenario.
    lsp_generator / hybrid_generator:
        Ready-to-use channel model helpers.
    rx_info:
        Optional metadata stored inside the PKL payload.
    """

    available: bool
    rt_params_np: dict[str, np.ndarray] | None
    los_flag_np: np.ndarray | None
    ue_coordinates: np.ndarray
    bs_coordinates: np.ndarray
    ut_panel_array: PanelArray | None
    bs_panel_array: PanelArray | None
    lsp_generator: LSPGenerator | None
    hybrid_generator: HybridClusterGenerator | None
    rx_info: Mapping[str, Any] | None = None


def load_rt_data_and_init_generator(
    carrier_frequency: float,
    *,
    scenario_state: Any | None = None,
    users: Iterable[Any] | None = None,
    bs_list: Iterable[Any] | None = None,
    rt_file_path: str | Path | None = None,
    o2i_model: str = "low",
    los_bs_indices: Iterable[int] | None = None,
) -> RayTracingLoaderResult:
    """Load ray-tracing tensors and configure hybrid cluster helpers.

    Parameters
    ----------
    carrier_frequency:
        Operating frequency in Hz used when creating Sionna scenarios.
    scenario_state:
        Optional :class:`ScenarioState` instance. When present it is used to
        derive UE/BS coordinates if ``users`` or ``bs_list`` are omitted.
    users / bs_list:
        Iterables describing the current topology. They are mainly consumed to
        derive coordinate matrices. When omitted the loader tries to pull them
        from ``scenario_state`` or falls back to metadata contained inside the
        dataset.
    rt_file_path:
        Path to the PKL file storing RT outputs. Defaults to
        ``channel_generator/hybrid_cluster/rt_outputs.pkl`` relative to the
        repository root.
    o2i_model:
        Outdoor-to-indoor setting used for the :class:`UMiScenario`.
    los_bs_indices:
        Iterable of BS indices that should be marked as LoS for all users.

    Returns
    -------
    RayTracingLoaderResult
        Dataclass containing the tensors, topology helper objects and a flag
        indicating whether the dataset was available.
    """

    if scenario_state is not None:
        users = getattr(scenario_state, "users", None) if users is None else users
        bs_list = getattr(scenario_state, "bs_list", None) if bs_list is None else bs_list

    rt_path = Path(rt_file_path) if rt_file_path is not None else _DEFAULT_RT_FILE

    try:
        import pickle

        with rt_path.open("rb") as fh:
            loaded = pickle.load(fh)
        if isinstance(loaded, Mapping):
            rt_params_raw = loaded
            rx_info = None
        elif isinstance(loaded, (list, tuple)) and len(loaded) >= 1:
            rt_params_raw = loaded[0]
            rx_info = loaded[1] if len(loaded) > 1 else None
        else:
            raise ValueError("Unsupported PKL format: expected dict or [dict, metadata]")
    except FileNotFoundError:
        LOGGER.warning("Ray-tracing dataset not found at %s", rt_path)
        user_count = _safe_len(users)
        bs_count = _safe_len(bs_list)
        return RayTracingLoaderResult(
            available=False,
            rt_params_np=None,
            los_flag_np=None,
            ue_coordinates=_user_coordinates(users, None, user_count),
            bs_coordinates=_bs_coordinates(bs_list, bs_count),
            ut_panel_array=None,
            bs_panel_array=None,
            lsp_generator=None,
            hybrid_generator=None,
            rx_info=None,
        )
    except Exception as exc:  # pragma: no cover - defensive logging branch
        LOGGER.warning("Failed to load ray-tracing dataset from %s: %s", rt_path, exc)
        user_count = _safe_len(users)
        bs_count = _safe_len(bs_list)
        return RayTracingLoaderResult(
            available=False,
            rt_params_np=None,
            los_flag_np=None,
            ue_coordinates=_user_coordinates(users, None, user_count),
            bs_coordinates=_bs_coordinates(bs_list, bs_count),
            ut_panel_array=None,
            bs_panel_array=None,
            lsp_generator=None,
            hybrid_generator=None,
            rx_info=None,
        )

    rt_params_np = {k: _to_batched_numpy(v) for k, v in rt_params_raw.items()}
    tau = rt_params_np.get("tau")
    if tau is None:
        raise KeyError("'tau' not found inside ray-tracing parameters")
    num_ut = tau.shape[1]
    num_bs = tau.shape[2]

    ue_coords = _user_coordinates(users, rx_info, num_ut)
    bs_coords = _bs_coordinates(bs_list, num_bs)

    los_flag_np = np.zeros((tau.shape[0], num_ut, num_bs), dtype=bool)
    if los_bs_indices is not None:
        for bs_idx in los_bs_indices:
            if 0 <= bs_idx < num_bs:
                los_flag_np[:, :, bs_idx] = True

    ut_panel_array = _build_default_ut_panel(carrier_frequency)
    bs_panel_array = _build_default_bs_panel(carrier_frequency)

    scenario = UMiScenario(
        carrier_frequency=carrier_frequency,
        o2i_model=o2i_model,
        ut_array=ut_panel_array,
        bs_array=bs_panel_array,
        direction="downlink",
    )
    lsp_generator = LSPGenerator(scenario)
    hybrid_generator = HybridClusterGenerator(scenario)

    return RayTracingLoaderResult(
        available=True,
        rt_params_np=rt_params_np,
        los_flag_np=los_flag_np,
        ue_coordinates=ue_coords,
        bs_coordinates=bs_coords,
        ut_panel_array=ut_panel_array,
        bs_panel_array=bs_panel_array,
        lsp_generator=lsp_generator,
        hybrid_generator=hybrid_generator,
        rx_info=rx_info,
    )


def _to_batched_numpy(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    arr = np.expand_dims(arr, axis=0)
    if arr.dtype.kind in {"f", "c"}:
        arr = arr.astype(np.float32)
    return arr


def _empty_coord_matrix(rows: int = 0) -> np.ndarray:
    return np.zeros((rows, 3), dtype=np.float32)


def _user_coordinates(users: Iterable[Any] | None, rx_info: Mapping[str, Any] | None, expected_ut: int) -> np.ndarray:
    coords: list[np.ndarray] = []
    for user in users or []:
        mobility = getattr(user, "mobility", None)
        if mobility is None:
            continue
        coordinate = getattr(mobility, "coordinate", None)
        if coordinate is None:
            continue
        coords.append(np.asarray(coordinate, dtype=np.float32).reshape(-1))
    if coords:
        return np.stack(coords, axis=0)
    if rx_info is not None and "rx_position" in rx_info:
        rx_position = np.asarray(rx_info["rx_position"], dtype=np.float32).reshape(1, 3)
        return np.repeat(rx_position, expected_ut, axis=0)
    return _empty_coord_matrix(expected_ut)


def _bs_coordinates(bs_list: Iterable[Any] | None, expected_bs: int) -> np.ndarray:
    coords: list[np.ndarray] = []
    for bs in bs_list or []:
        coordinate = getattr(bs, "coordinate", None)
        if coordinate is None:
            continue
        coords.append(np.asarray(coordinate, dtype=np.float32).reshape(-1))
    if coords:
        stacked = np.stack(coords, axis=0)
        if stacked.shape[0] >= expected_bs:
            return stacked[:expected_bs]
        padding = _empty_coord_matrix(expected_bs - stacked.shape[0])
        return np.vstack([stacked, padding])
    return _empty_coord_matrix(expected_bs)


def _build_default_ut_panel(carrier_frequency: float) -> PanelArray:
    return PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=2,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
    )


def _build_default_bs_panel(carrier_frequency: float) -> PanelArray:
    return PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="dual",
        polarization_type="VH",
        antenna_pattern="omni",
        carrier_frequency=carrier_frequency,
    )


def _safe_len(items: Iterable[Any] | None) -> int:
    if items is None:
        return 0
    try:
        return int(len(items))  # type: ignore[arg-type]
    except TypeError:
        return 0


__all__ = [
    "RayTracingLoaderResult",
    "load_rt_data_and_init_generator",
]
