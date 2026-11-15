"""Stacking 32조합 runner: 팀원 JointPolicy와의 동등성, ④ 그리퍼 제외, 집계 형태."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs.config import REPO
from fm_constraint.joint import JointPolicy, JointPotential
from fm_constraint.joint_combo import JointComboPolicy, combo_flags, rtc_target
from fm_constraint import stacking_experiment as se


@pytest.fixture
def settings():
    return json.loads((REPO / 'configs/stacking_combo29.json').read_text())


@pytest.fixture
def potential(tmp_path):
    support = np.random.default_rng(42).normal(size=(12, 7)) * 0.1
    path = tmp_path / 'cache.npz'
    np.savez(path, support=support, safe_anchors=support[:4], joint_center=np.zeros(7),
             joint_scale=np.ones(7), bandwidth=np.array(0.15))
    return JointPotential(path, 'cpu')


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((31,), -0.2))

    def forward(self, x, t):
        # 채널 사이를 섞어 그리퍼 target이 다른 채널에 영향을 줄 수 있는 모델
        return self.w * x + 0.05 * x.roll(1, dims=-1) + 0.01 * t


def fake_base():
    def condition(x, cond, action_dim):
        x = x.clone()
        x[:, 0, action_dim:] = cond[0]
        return x
    return SimpleNamespace(center=torch.linspace(-0.01, 0.01, 31), scale=torch.linspace(0.07, 0.13, 31),
                           device='cpu', apply_conditioning=condition, model=Model(), act_dim=8, horizon=16)


def inputs(settings, seed=1729):
    q = (np.array(settings['joint_low_rad']) + settings['joint_high_rad']) / 2 + np.arange(7) * 1e-9
    obs = np.r_[q, 0.08, np.zeros(15)].astype(np.float32)
    noise = torch.randn(1, 16, 31, generator=torch.Generator().manual_seed(seed))
    return obs, np.full(7, 1e-4), np.asarray([q - 2e-4, q - 1e-4, q]), noise


@pytest.mark.parametrize('number,enabled', [(0, False), (4, False), (29, True)])
def test_matches_teammate_joint_policy(settings, potential, number, enabled):
    """조합 0·4(계획 단계 off)는 baseline arm, 29는 팀원 joint_combo29 arm과 비트 단위로 같다."""
    base = fake_base()
    obs, prev, hist, noise = inputs(settings)
    ours = JointComboPolicy(base, settings, potential, combo_flags(number)).plan(obs, prev, hist, noise)
    theirs = JointPolicy(base, settings, potential if enabled else None, enabled).plan(obs, prev, hist, noise)
    assert ours is not None and theirs is not None
    assert np.array_equal(ours, theirs)


def test_rtc_ignores_gripper_target_but_follows_joints(settings):
    base = fake_base()
    obs, prev, hist, noise = inputs(settings)
    policy = JointComboPolicy(base, settings, None, combo_flags(2))
    suffix = np.zeros((8, 8))
    a = policy.plan(obs, prev, hist, noise, rtc_suffix=suffix)
    assert policy.last_report['rtc_guided']
    grip = suffix.copy()
    grip[:, 7] = 0.5
    b = policy.plan(obs, prev, hist, noise, rtc_suffix=grip)
    joint = suffix.copy()
    joint[:, :7] = 0.01
    c = policy.plan(obs, prev, hist, noise, rtc_suffix=joint)
    plain = policy.plan(obs, prev, hist, noise, rtc_suffix=None)
    assert np.array_equal(a, b)                       # 그리퍼 target은 아무 영향이 없다
    assert not np.allclose(a[:, :7], c[:, :7])        # 관절 target은 계획을 바꾼다
    assert not np.array_equal(a, plain)
    _, w = rtc_target(base, suffix)
    assert not w[..., 7].count_nonzero() and w[0, :8, 0].min() > 0 and not w[0, 8:].count_nonzero()
    assert all(p.requires_grad for p in base.model.parameters())


def test_lipo_without_projection_and_no_rejection(settings):
    base = fake_base()
    obs, prev, hist, noise = inputs(settings)
    policy = JointComboPolicy(base, settings, None, combo_flags(1))
    out = policy.plan(obs, prev, hist, noise)
    r = policy.last_report
    assert out is not None and r['lipo']['status'] == 'ok' and r['projection_calls'] == 0
    # 보정량은 float64 계획 기준(조합 0은 float32 경로라 직접 비교하면 반올림 오차가 누적된다)
    assert r['lipo']['correction_max_rad'] <= settings['lipo_bound_rad'] + 1e-12
    assert r['lipo']['jerk_cost_after_rad2'] <= r['lipo']['jerk_cost_before_rad2']
    raw = JointComboPolicy(base, settings, None, combo_flags(0)).plan(obs, prev, hist, noise)
    np.testing.assert_allclose(out[:, 7], raw[:, 7], atol=1e-6)   # 그리퍼는 LiPo가 바꾸지 않는다


def test_pf_requires_cache(settings):
    with pytest.raises(ValueError):
        JointComboPolicy(fake_base(), settings, None, combo_flags(8))


def test_context_sha_and_noise_seed():
    ctx = ([np.array([0.4, -0.2, 10.], np.float32), np.array([1., 0, 0, 0])],) * 4
    assert len(se.context_sha(ctx)) == 64
    # 팀원 eval_stacking의 규칙과 같다
    import hashlib
    payload = json.dumps(['stacking', 'test_007', 1], separators=(',', ':'))
    assert se.noise_seed(7, 1) == int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') % (2 ** 63 - 1)


def episode(success, g1, g2, viol, steps, mode=0, failure=None):
    return dict(success=success, success_1=g1, success_2=g2, constraint_violation_steps=viol,
                constraint_evaluated_steps=steps, action_change_rad=0.01, command_jerk_rad_s3=None,
                hand_jerk_m_s3=2.0, boundary_jump_rad_s=0.1, mode=mode, failure=failure, plan_ms_mean=50.)


def test_aggregate_rates():
    rows = [se.context_row([episode(True, True, True, 1, 100), episode(False, True, False, 0, 300, failure='infeasible'),
                            episode(False, False, False, 3, 100)]),
            se.context_row([episode(True, True, True, 0, 100, mode=1)] * 3)]
    v = se.aggregate(np.array(rows, dtype=float))
    assert v.shape == (len(se.METRICS),)
    assert v[0] == pytest.approx(100 * (1 / 3 + 1) / 2)
    assert v[1] == pytest.approx(100 * (2 / 3 + 1) / 2)
    assert v[3] == pytest.approx(100 * 4 / 800)        # step 분모를 합친다
    assert np.isnan(v[5])                                # 정의되지 않은 jerk는 0으로 채우지 않는다
    assert v[9] == pytest.approx(100 * (1 / 3) / 2)
