import math

import numpy as np

from phase_causal.config import TaskConfig
from phase_causal.data import generate_sample


def test_delay_contains_no_phase_input_and_target_is_exact():
    cfg = TaskConfig(total_steps=40, cue_steps=10, phase_channels=8)
    phi = 0.3
    omega = 2.0 * math.pi / 20.0

    x, target, metadata = generate_sample(
        phi=phi,
        omega=omega,
        cfg=cfg,
        rng=np.random.default_rng(7),
    )

    assert x.shape == (40, 10)
    assert np.count_nonzero(x[cfg.cue_steps :, : cfg.phase_channels]) == 0
    t = np.arange(cfg.total_steps, dtype=np.float32)
    expected = np.stack((np.cos(phi + omega * t), np.sin(phi + omega * t)), axis=-1)
    np.testing.assert_allclose(target, expected, atol=1e-6)
    assert metadata == {"phi": phi, "omega": omega}


def test_marker_channels_identify_cue_and_delay_boundaries():
    cfg = TaskConfig(total_steps=30, cue_steps=8, phase_channels=4)
    x, _, _ = generate_sample(0.0, 0.1, cfg, np.random.default_rng(1))

    assert x[0, cfg.phase_channels] == 1.0
    assert x[cfg.cue_steps, cfg.phase_channels + 1] == 1.0
    assert x[:, cfg.phase_channels :].sum() == 2.0

