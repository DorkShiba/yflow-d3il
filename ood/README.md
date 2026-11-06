# OOD 초기 조건 생성·검증 도구

배포 데이터와 사용법은 [데이터셋 README](../data/ood_sas_v1_20261106/README.md)를 참고하세요.

- `generate.py`: 결정적 후보 생성, 조건 정의, 방법 간 paired 관측 노이즈.
- `verify.py`: SHA256, 사례 수, 영역 밖 거리, 분리 조건, paired seed 검사. 시뮬레이션은 실행하지 않습니다.
- `validate_sim.py`: 실제 MuJoCo 초기화/IK/충돌/안정화 스크리닝. 태스크 성공이나 controller 연결 검증은 아닙니다.
- `summarize_sim.py`: 초기화 결과를 통과 ID와 탈락 사유로 집계합니다.

원본 후보는 변경하지 말고 물리 검사 결과를 별도 파일로 보존하세요. 학습이나 정책 성능 평가는 이 도구에 포함하지 않습니다.
