# LiPo Inserting 비교 결과

2026-11-04 · RTX 3090 · 동일 EMA 체크포인트 · 검증 분할 초기 배치 평가

**LiPo는 실제 손 jerk RMS를 17.3%, 경계 속도 변화를 58.9% 줄였다. 성공률은 263/328에서 256/328로 2.13%p 낮게 관측됐다.** 성공률 차이의 paired context-bootstrap 95% CI는 −7.32~+3.05%p로 0을 포함한다. 따라서 smoothness 개선은 확인되지만 성공률 개선·유지 또는 비열등성은 입증되지 않았다. Pushing에서 사용한 보정 파라미터를 변경 없이 적용한 결과이며, 이 결과에 맞춘 재튜닝은 수행하지 않았다.

고정 설정: context 82개 × 4회, seed 0, 16스텝 생성·8스텝 실행, Euler 10스텝, paired noise.
LiPo: blend 4스텝, blend bound 2 mm, path bound 0.3 mm. 평가 결과에 따른 파라미터 조정 없음.

| 지표 | Baseline | Blend only | LiPo |
|---|---:|---:|---:|
| 성공률 | 263/328 (80.18%) | 264/328 (80.49%) | 256/328 (78.05%) |
| desired_jerk_rms_m_s3 | 114.847 | 111.746 | 55.3817 |
| actual_jerk_rms_m_s3 | 44.4486 | 46.9807 | 36.7616 |
| seam_velocity_jump_rms_m_s | 0.191219 | 0.0856953 | 0.0786106 |
| actual_seam_jerk_rms_m_s3 | 58.9125 | 43.1381 | 38.7281 |
| tracking_gap_mean_m | 0.0213282 | 0.0216406 | 0.022851 |
| duration_s | 46.7943 | 46.0152 | 47.4005 |
| plan_ms | 89.265 | 89.287 | 89.163 |
| postprocess_ms | 0.018 | 0.051 | 1.885 |
| qp_ms | — | — | 1.815 |

## LiPo − Baseline: paired context-bootstrap 95% CI

| 지표 | 평균 차이 | 95% CI | 상대 변화 |
|---|---:|---:|---:|
| success_rate | -2.1341 %p | [-7.3171, 3.0488] %p | -2.66% |
| desired_jerk_rms_m_s3 | -59.465 | [-61.317, -57.745] | -51.78% |
| actual_jerk_rms_m_s3 | -7.6869 | [-8.5156, -6.8359] | -17.29% |
| seam_velocity_jump_rms_m_s | -0.11261 | [-0.11683, -0.10877] | -58.89% |
| actual_seam_jerk_rms_m_s3 | -20.184 | [-21.282, -19.134] | -34.26% |
| tracking_gap_mean_m | 0.0015228 | [-0.0014808, 0.0049269] | 7.14% |
| duration_s | 0.6062 | [-1.2711, 2.4525] | 1.30% |

## 해석 범위

- 실제 손 지표는 D3IL 시뮬레이터의 XY 위치를 35 ms마다 샘플링한 유한차분이다. 물리 로봇이나 substep 고주파 진동의 개선을 입증하지 않는다.
- 이 결과는 검증 시연 82개 초기 배치의 비교다. OOD·지연·다른 태스크는 평가하지 않았다.
- 성공률 CI가 0을 포함하면 동일 성능 또는 비열등성을 입증한 것으로 해석하지 않는다.
- 전체 episode smoothness에는 성공 여부·길이·경로 차이가 영향을 줄 수 있다. 경계 지표, 이동량, 성공률을 함께 본다.
- 시간은 14개 평가 프로세스가 같은 GPU를 공유한 조건이다. 단독 배포 지연 벤치마크가 아니다. QP 시간은 후처리 시간에 포함된다.
- 분할 실행은 global context index로 난수를 고정하고, context별 raw rollout과 QP 진단을 합쳐 분석했다.

## 근거 파일

- `protocol.json`: 사전 고정 설정과 소스·체크포인트 해시
- `comparison.png`: context-bootstrap CI를 포함한 비교 그림
- `summary.json`: 핵심 수치와 paired 비교 결과
- `artifacts/main/combined/{none,blend,lipo}/`: 원본 궤적·episode/plan별 지표
- `artifacts/main/commands.json`: 모든 shard 실행 명령
- `artifacts/main/COMPLETE.json`: 완료 시각·실행 시간
- `artifacts/environment.txt`, `artifacts/source.tar.gz`: 환경과 실행 소스 스냅샷

## 원본 검증

- 원본 984개 rollout에서 별도 4점 차분식으로 jerk를 재계산했다. 저장 지표와 최대 차이는 3.06e-13이다.
- LiPo QP 55,637회가 모두 정상 종료했고 보정 한도와 목적함수 비증가를 확인했다.
- 소스·체크포인트는 사전 protocol.json과 일치하며, 내려받은 combined 결과 9개 파일의 SHA256을 확인했다.
- smoke와 본 평가의 공통 초기 rollout은 세 비교군 모두 목표·실제 손 위치 배열이 정확히 일치했다.
- 자세한 검증은 verification.json에 보존했다. 본 평가 소요 시간은 5,767초(약 96분)였다.
