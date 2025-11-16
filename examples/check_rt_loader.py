#!/usr/bin/env python3
"""Sanity check script for :mod:`sionna.sys.topology.ray_tracing_loader`."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "sionna" not in sys.modules:
    spec = importlib.util.spec_from_file_location("sionna", ROOT / "__init__.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError("Unable to locate the sionna package entry point")
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
    sys.modules["sionna"] = module
    spec.loader.exec_module(module)

from sionna.sys.topology.ray_tracing_loader import load_rt_data_and_init_generator


def main() -> None:
    result = load_rt_data_and_init_generator(
        carrier_frequency=3.5e9,
        los_bs_indices=[22],
    )

    if not result.available:
        print("⚠️  Ray-tracing dataset missing. Nothing to validate.")
        return

    tau = result.rt_params_np["tau"]
    a_re = result.rt_params_np["a_re"]

    assert tau.ndim == 4, f"Unexpected tau shape: {tau.shape}"
    assert a_re.ndim == 6, f"Unexpected a_re shape: {a_re.shape}"
    assert tau.shape[0] == 1, "Batched tensors must include B=1"
    assert a_re.shape[0] == 1, "Batched tensors must include B=1"
    assert tau.shape[1] == result.ue_coordinates.shape[0], "UT dimension mismatch"
    assert tau.shape[2] == result.bs_coordinates.shape[0], "BS dimension mismatch"
    assert result.los_flag_np.shape[:3] == tau.shape[:3], "LoS mask shape mismatch"

    print("✅ Ray-tracing loader sanity check passed")
    print(f"    tau shape: {tau.shape}")
    print(f"    a_re shape: {a_re.shape}")
    print(f"    los flag shape: {result.los_flag_np.shape}")
    print(f"    UE coordinates: {result.ue_coordinates.shape}")
    print(f"    BS coordinates: {result.bs_coordinates.shape}")


if __name__ == "__main__":
    main()
