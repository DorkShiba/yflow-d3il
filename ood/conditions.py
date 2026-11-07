"""OOD 조건 정의와 seed 규칙 (PLAN 3.1절, 2026-11-07 결정).

조건 강도는 구현 1(feature/ood-benchmark), seed 규칙은 구현 2(ood_sas_v1)를 따른다.

    A  블록 초기 위치   모든 블록을 시연 초기 범위 밖 d m로 이동 (가장 가까운 가능한 면)
    B  로봇 시작 위치   손을 고정 시작점에서 d m 떨어진 임의 방향으로 이동 (시뮬레이터 검증)
    C  관측 노이즈      정책이 보는 블록 x, y (σ_pos m)와 yaw (σ_yaw deg)에 함께 Gaussian
    D  실행 중 push     push_step 이후 손에 가장 가까운 블록을 d m 순간이동
    E  추론 지연        다음 계획을 d step 전 관측으로 세우고 새 chunk의 index d부터 실행

조건: id, {A,B,C}_{weak,mid,strong}, {AB,AC,BC,ABC}_mid, D_{...}, E_{...} (20개).
D, E는 단일 요인만 둔다.

seed는 모두 ``seed_for``(SHA-256 해시)로 만든다. 키를 요인 단위로 잡아,
같은 context에서는 강도·조합이 달라도 같은 난수를 쓴다(common random numbers).
관측 노이즈는 (case seed, rollout, step, entity, component)마다 독립 값이라
재계획 횟수·호출 순서가 방법마다 달라도 같은 시점의 노이즈가 같다.
"""
import hashlib
import random

ROOT_SEED = 20261106
POLICY_REPEATS = 4
NAMES = ('weak', 'mid', 'strong')
FACTORS = 'ABCDE'

# (weak, mid, strong). A, B, D: m; C: (σ_pos m, σ_yaw deg); E: control steps.
# A·B·C 수치는 구현 1(make_ood_dataset.LEVELS)과 같다. A = 블록 변 b의 0.5/1/2배, C = b의 5/10/20 %.
LEVELS = {
    'pushing': {'A': (0.025, 0.05, 0.10), 'B': (0.02, 0.05, 0.10),
                'C': ((0.0025, 2.5), (0.005, 5.0), (0.010, 10.0)),
                'D': (0.02, 0.04, 0.08), 'E': (1, 2, 4)},
    'inserting': {'A': (0.025, 0.05, 0.10), 'B': (0.02, 0.05, 0.10),
                  'C': ((0.0025, 2.5), (0.005, 5.0), (0.010, 10.0)),
                  'D': (0.02, 0.04, 0.08), 'E': (1, 2, 4)},
}
# D: 'mid'는 chunk 실행 도중(남은 행동 execute/2개)에 밀어 남은 행동이 바뀐 상태를 모른 채 실행된다.
#    'seam'은 재계획 직전에 민다.
PUSH_PHASE = 'mid'


def seed_for(*parts):
    """구현 2의 해시 seed. 순서가 있는 값 목록 → 64-bit 정수."""
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big')


def gauss(seed, *key, sigma):
    """키마다 독립인 Gaussian 값. sigma=0이면 정확히 0이다."""
    if not sigma:
        return 0.0
    return random.Random(seed_for(seed, *key)).gauss(0.0, sigma)


def conditions():
    """[(이름, {요인: 강도 index})] 고정 순서."""
    out = [('id', {})]
    for factor in 'ABC':
        out += [(f'{factor}_{name}', {factor: k}) for k, name in enumerate(NAMES)]
    for combo in ('AB', 'AC', 'BC', 'ABC'):
        out.append((f'{combo}_mid', {factor: 1 for factor in combo}))
    for factor in 'DE':
        out += [(f'{factor}_{name}', {factor: k}) for k, name in enumerate(NAMES)]
    return out


def factor_values(task, selected):
    """선택된 요인의 실제 수치. 선택되지 않은 요인은 None."""
    levels = LEVELS[task]
    values = dict.fromkeys(FACTORS)
    for factor, k in selected.items():
        level = {'level': NAMES[k]}
        if factor == 'A':
            values['A'] = {'shift_m': levels['A'][k], **level}
        elif factor == 'B':
            values['B'] = {'dist_m': levels['B'][k], **level}
        elif factor == 'C':
            pos, yaw = levels['C'][k]
            values['C'] = {'pos_m': pos, 'yaw_deg': yaw, **level}
        elif factor == 'D':
            values['D'] = {'dist_m': levels['D'][k], 'phase': PUSH_PHASE, **level}
        else:
            values['E'] = {'delay_steps': levels['E'][k], **level}
    return values


def context_seeds(task, context_id):
    """context 하나의 요인별 seed. 모든 조건에서 공유한다."""
    return {'B': seed_for(ROOT_SEED, task, context_id, 'B'),
            'C': seed_for(ROOT_SEED, task, context_id, 'C'),
            'D': seed_for(ROOT_SEED, task, context_id, 'D'),
            'policy': [seed_for(ROOT_SEED, task, context_id, 'policy', j) for j in range(POLICY_REPEATS)]}
