# 포함한 외부 코드의 출처

## FM 모델과 학습·적분 도구

- 원본 저장소: https://github.com/azizanlab/HardFlow
- 기준 브랜치: `d3il`
- 기준 커밋: `a85105080dd0a28dca76b49ac0414f275de146ed`
- 이관 전 실제 파일의 SHA-256: `fm_baseline/models/provenance.json`

현재 실험에서 사용하는 다음 부분만 저장소 내부로 가져왔다.

| 내부 파일 | 원본 파일·범위 | 조정 사항 |
|---|---|---|
| `fm_baseline/models/temporal_unet.py` | `hardflow/models_flow/unet.py`의 TemporalUnet 및 구성 블록 | 사용하지 않는 loss·diffusion 도구·WrappedFlowUnet·utils import 제외. 파라미터 이름과 계산 순서 유지 |
| `fm_baseline/models/flow_matching.py` | `hardflow/models_flow/flow_matcher.py` | CFM 학습·조건 고정·난수 생성 순서 유지 |
| `fm_baseline/models/ode.py` | `hardflow/models_flow/flow_policy.py`의 ConditionedODESolver | 현재 호출하는 value=None, Euler 경로 및 선택적 chain 반환만 유지. 미사용 value-guidance 경로 제외 |

이관 당시 원본 checkout에는 별도의 LICENSE/COPYING/NOTICE 파일이 없었다. 여기서는 원본의 저작자·저장소·커밋 출처를 보존하며 원본 코드에 새로운 라이선스를 임의로 부여하지 않는다.

GuideFlow, SafeFlow, UniConFlow, YFlow 및 HardFlow의 별도 제약 solver·FlowPolicy는 포함하지 않았다. 이 저장소의 Projection·PF2MP·실행 필터는 `fm_constraint/`의 자체 구현을 사용한다. 외부 HardFlow 설치·checkout은 런타임과 테스트에 필요하지 않다. D3IL 시뮬레이터 의존성은 별도로 유지된다.
