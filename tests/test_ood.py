"""OOD 조건·seed 규칙·실행 훅 (시뮬레이터 없이)."""
import json
from pathlib import Path

import numpy as np
import pytest

from ood.apply import Episode, context_from_json, context_to_json, load_cases
from ood.conditions import conditions, context_seeds, factor_values, gauss, seed_for

REPO = Path(__file__).resolve().parents[1]


def test_condition_list_is_fixed():
    names = [name for name, _ in conditions()]
    assert len(names) == 20 and names[0] == 'id'
    assert names[10:14] == ['AB_mid', 'AC_mid', 'BC_mid', 'ABC_mid']
    assert names[-6:] == ['D_weak', 'D_mid', 'D_strong', 'E_weak', 'E_mid', 'E_strong']


def test_strengths_follow_implementation_1():
    v = factor_values('pushing', dict(conditions())['ABC_mid'])
    assert v['A']['shift_m'] == 0.05 and v['B']['dist_m'] == 0.05
    assert (v['C']['pos_m'], v['C']['yaw_deg']) == (0.005, 5.0)   # xy와 yaw를 함께 적용
    assert v['D'] is None and v['E'] is None
    assert factor_values('inserting', {'E': 2})['E']['delay_steps'] == 4


def test_seed_rule_is_hash_based_and_shared_across_conditions():
    a, b = context_seeds('pushing', 'test_000'), context_seeds('pushing', 'test_001')
    assert a == context_seeds('pushing', 'test_000')
    assert a['C'] != b['C'] and len(set(a['policy'])) == 4
    assert seed_for(1, 'x') == seed_for('1', 'x')       # 구현 2와 같은 '|'.join 규칙


def test_noise_value_is_independent_of_call_order():
    first = [gauss(7, 0, step, 0, 'x', sigma=0.01) for step in range(5)]
    second = [gauss(7, 0, step, 0, 'x', sigma=0.01) for step in reversed(range(5))][::-1]
    assert first == second and gauss(7, 0, 0, 0, 'x', sigma=0.0) == 0.0


def test_episode_observation_noise_and_push_timing():
    case = {'factors': {}, 'noise': {'pos_m': 0.01, 'yaw_deg': 5.0, 'seed': 3},
            'push': {'dist_m': 0.04, 'phase': 'mid', 'step': 10, 'seed': 4}, 'delay_steps': 0}
    boxes = [(np.array([0.4, 0.0, 0.0]), None, 0.0)]
    ep = Episode(case, rollout=1, execute_steps=8)
    noisy = ep.observe_boxes(boxes, step=5)
    assert noisy[0][0][2] == 0.0 and not np.allclose(noisy[0][0][:2], [0.4, 0.0])
    assert np.array_equal(noisy[0][0], Episode(case, rollout=1).observe_boxes(boxes, step=5)[0][0])
    assert boxes[0][0][0] == 0.4                          # 실제 상태는 그대로
    assert not ep.push_due(9, 4) and not ep.push_due(10, 0) and ep.push_due(10, 4)
    off = Episode(None, rollout=0)
    assert off.observe_boxes(boxes, 0) is boxes and not off.push_due(100, 0) and off.robot_start is None


def test_delay_must_leave_room_in_the_chunk():
    with pytest.raises(ValueError):
        Episode({'delay_steps': 8}, rollout=0, execute_steps=8)


def test_context_json_round_trip():
    ctx = [np.array([0.4, -0.1, 30.0]), np.array([1.0, 0.0, 0.0, 0.0])]
    back = context_from_json(json.loads(json.dumps(context_to_json(ctx))))
    assert all(np.array_equal(a, b) for a, b in zip(ctx, back))


@pytest.mark.skipif(not (REPO / 'data/ood/manifest.json').exists(), reason='OOD data not generated')
def test_generated_dataset_verifies():
    from ood.verify import verify
    manifest, errors = verify(REPO / 'data/ood')
    assert not errors, errors[:5]
    cases, info = load_cases(REPO / 'data/ood', 'pushing', 'D_mid')
    assert len(cases) == manifest['tasks']['pushing']['n_contexts'] and info['condition'] == 'D_mid'
