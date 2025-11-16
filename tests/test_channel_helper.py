"""Regression tests for :mod:`scenario_generator.channel_helper`."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import tensorflow as tf

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(ROOT))

from scenario_generator.BS import BS
from scenario_generator.channel_helper import OFDMNumerology, get_cfr_from_ray_tracing
from scenario_generator.scenario_main import ScenarioState
from scenario_generator.user import Mobility, User


class DummyScenario:
    """Minimal object exposing :py:meth:`set_topology`."""

    def __init__(self):
        self.last_kwargs = None

    def set_topology(self, **kwargs):  # pragma: no cover - simple storage
        self.last_kwargs = kwargs


class DummyHybridGenerator:
    """Return deterministic channels for regression tests."""

    def __init__(self, num_bs: int, numerology: OFDMNumerology):
        self.num_bs = num_bs
        self.numerology = numerology
        self.calls = 0

    def __call__(
        self,
        *,
        user,
        sample_times,
        normalize,
        num_subcarriers,
        subcarrier_spacing,
    ):
        del user, sample_times, normalize, num_subcarriers, subcarrier_spacing
        self.calls += 1
        payload = (
            tf.ones(
                [
                    1,
                    1,
                    self.numerology.num_ut_ant,
                    self.num_bs,
                    self.numerology.num_bs_ant,
                    self.numerology.num_ofdm_symbols,
                    self.numerology.num_subcarriers,
                ],
                dtype=tf.complex64,
            )
            * self.calls
        )
        rt_mask = tf.ones([1, 1, self.num_bs], dtype=tf.bool)
        return payload, rt_mask


def _make_user(user_id: str, coord) -> User:
    mobility = Mobility(
        coordinate=np.asarray(coord),
        velocity=np.zeros(3),
        orientation=np.zeros(3),
    )
    return User(user_id=user_id, mobility=mobility)


def test_get_cfr_from_ray_tracing_stacks_channels(tmp_path):
    numerology = OFDMNumerology(
        num_ut_ant=2,
        num_bs_ant=1,
        num_ofdm_symbols=2,
        num_subcarriers=4,
        subcarrier_spacing=15e3,
    )
    state = ScenarioState()
    state.users = [
        _make_user("ue-0", [0.0, 0.0, 1.5]),
        _make_user("ue-1", [5.0, 0.0, 1.5]),
    ]
    rt_file = tmp_path / "ue0.pkl"
    rt_file.write_bytes(b"dummy")
    state.users[0].pkl = str(rt_file)
    state.bs_list = [BS(bs_id="bs0", coordinate=[0.0, 0.0, 25.0], orientation=[0.0, 0.0, 0.0])]

    scenario = DummyScenario()
    generator = DummyHybridGenerator(num_bs=len(state.bs_list), numerology=numerology)
    rng = tf.random.Generator.from_seed(13)

    h_slot, metadata = get_cfr_from_ray_tracing(
        state,
        numerology,
        scenario,
        generator,
        rng=rng,
    )

    assert generator.calls == 1
    assert h_slot.shape == (
        len(state.users),
        numerology.num_ut_ant,
        len(state.bs_list),
        numerology.num_bs_ant,
        numerology.num_ofdm_symbols,
        numerology.num_subcarriers,
    )
    np.testing.assert_allclose(
        h_slot[0].numpy(),
        np.ones((2, 1, 1, 2, 4), dtype=np.complex64),
    )
    assert not np.allclose(h_slot[1].numpy(), 0.0)
    assert metadata[0]["from_ray_tracing"] is True
    assert metadata[0]["rt_file"] == str(rt_file)
    assert metadata[1]["from_ray_tracing"] is False
    assert scenario.last_kwargs["ut_loc"].shape == (1, len(state.users), 3)
