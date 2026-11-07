# LiPo Pushing 비교 결과

2026-11-04 · RTX 3090 · 동일 EMA 체크포인트 · ID 평가

**이번 고정 설정에서 LiPo는 실제 손 jerk RMS를 25.3%, 경계 속도 변화를 59.6% 줄였다. 성공률은 106/120에서 115/120으로 7.5%p 상승했다.** 성공률 차이의 paired context-bootstrap 95% CI는 +1.67~+14.17%p다. Blending만으로는 전체 실제 손 jerk RMS가 줄지 않았으므로, QP를 추가한 비교군의 smoothness 개선이 확인된다.

고정 설정: context 30개 × 4회, seed 0, 16스텝 생성·8스텝 실행, Euler 10스텝, paired noise.
LiPo: blend 4스텝, blend bound 2 mm, path bound 0.3 mm. 평가 결과에 따른 파라미터 조정 없음.

| 지표 | Baseline | Blend only | LiPo |
|---|---:|---:|---:|
| 성공률 | 106/120 (88.33%) | 114/120 (95.00%) | 115/120 (95.83%) |
| desired_jerk_rms_m_s3 | 98.0158 | 90.5138 | 42.4991 |
| actual_jerk_rms_m_s3 | 36.2531 | 36.7837 | 27.0757 |
| seam_velocity_jump_rms_m_s | 0.160414 | 0.0706218 | 0.0647653 |
| actual_seam_jerk_rms_m_s3 | 49.1505 | 33.7268 | 28.5616 |
| tracking_gap_mean_m | 0.0153047 | 0.0151996 | 0.0151997 |
| duration_s | 8.66746 | 7.95346 | 8.07158 |
| plan_ms | 63.734 | 63.910 | 63.537 |
| postprocess_ms | 0.018 | 0.050 | 1.841 |
| qp_ms | — | — | 1.773 |

## LiPo − Baseline: paired context-bootstrap 95% CI

| 지표 | 평균 차이 | 95% CI | 상대 변화 |
|---|---:|---:|---:|
| success_rate | 7.5 %p | [1.6667, 14.167] %p | 8.49% |
| desired_jerk_rms_m_s3 | -55.517 | [-58.53, -52.782] | -56.64% |
| actual_jerk_rms_m_s3 | -9.1774 | [-9.987, -8.365] | -25.31% |
| seam_velocity_jump_rms_m_s | -0.095649 | [-0.1023, -0.090056] | -59.63% |
| actual_seam_jerk_rms_m_s3 | -20.589 | [-22.168, -19.08] | -41.89% |
| tracking_gap_mean_m | -0.00010506 | [-0.0030866, 0.0033125] | -0.69% |
| duration_s | -0.59587 | [-1.0022, -0.196] | -6.87% |

## 해석 범위

- 실제 손 지표는 D3IL 시뮬레이터의 XY 위치를 35 ms마다 샘플링한 유한차분이다. 물리 로봇이나 substep 고주파 진동의 개선을 입증하지 않는다.
- 이 결과는 공식 test context 앞 30개의 ID 비교다. OOD·지연·다른 태스크는 평가하지 않았다.
- 성공률 CI가 0을 포함하면 동일 성능 또는 비열등성을 입증한 것으로 해석하지 않는다.
- 전체 episode smoothness에는 성공 여부·길이·경로 차이가 영향을 줄 수 있다. 경계 지표, 이동량, 성공률을 함께 본다.
- 시간은 9개 평가 프로세스가 같은 GPU를 공유한 조건이다. 단독 배포 지연 벤치마크가 아니다. QP 시간은 후처리 시간에 포함된다.
- 분할 실행은 global context index로 난수를 고정하고, context별 raw rollout과 QP 진단을 합쳐 분석했다.

## 근거 파일

- `protocol.json`: 사전 고정 설정과 소스·체크포인트 해시
- `comparison.png`: context-bootstrap CI를 포함한 비교 그림
- `summary.json`: 핵심 수치와 paired 비교 결과
- `artifacts/main/combined/{none,blend,lipo}/`: 원본 궤적·episode/plan별 지표
- `artifacts/main/commands.json`: 모든 shard 실행 명령
- `artifacts/main/COMPLETE.json`: 완료 시각·실행 시간
- `artifacts/environment.txt`, `artifacts/source-main.tar.gz`: 환경과 실행 소스 스냅샷

## 원본 검증

- 360개 raw rollout에서 jerk를 별도 4점 차분식으로 다시 계산했다. 저장된 RMS와의 최대 차이는 7.25e-13이었다.
- LiPo QP 3,507회 모두 정상 종료했고, 각 청크의 QP 목적함수가 증가하지 않았으며 보정 한도를 지켰다.
- 최대 QP 보정은 2 mm 이내였다. Blending까지 포함한 raw 정책 대비 최대 위치 변위는 16.59 mm로, QP bound와 구분해야 한다.
- 순차 smoke 평가의 첫 6개 rollout과 병렬 본 평가의 해당 rollout은 세 비교군 모두 목표·실제 위치 배열이 정확히 일치했다.
- 실행 소스·체크포인트는 사전 `protocol.json`과 일치했고, 내려받은 combined 결과 9개 파일의 SHA256도 확인했다.
- `verification.json`에 검증 수치를 저장했다. 본 평가 벽시계 시간은 약 10분이었다.
