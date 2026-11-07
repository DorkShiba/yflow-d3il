"""Summarize completed initialization screens without promoting rollout readiness."""
import argparse,collections,hashlib,json
from pathlib import Path

def summarize(root,inputs):
    root=Path(root);inputs=Path(inputs);summary={'scope':'MuJoCo static initialization, start-pose IK and 0.6 s fixed-robot object settling','not_validated':['closed-loop policy performance','controller state synchronization','workspace and object-to-goal path feasibility','empirical train support'],'tasks':{},'rollout_ready_cases':0}
    rejected=[];accepted=[]
    for task in ['sorting2','sorting4','sorting6','stacking','avoiding']:
        complete=json.loads((root/(task+'_complete.json')).read_text())
        rows=[json.loads(s) for s in (root/(task+'_results.jsonl')).read_text().splitlines()]
        candidates=[json.loads(s) for s in (inputs/(task+'.jsonl')).read_text().splitlines()]
        assert len(rows)==len(candidates)==complete['cases']
        assert [x['id'] for x in rows]==[x['id'] for x in candidates]
        counts=collections.Counter();by=collections.defaultdict(collections.Counter);reasons=collections.Counter()
        for r in rows:
            counts[r['initialization_screen']]+=1;by[f"{r['family']}/L{r['level']}"][r['initialization_screen']]+=1
            assert r['rollout_ready'] is False
            if r['initialization_screen']=='pass':accepted.append(r['id'])
            else:rejected.append({'id':r['id'],'reasons':r['reasons']});reasons.update(r['reasons'])
        assert dict(counts)==complete['counts']
        summary['tasks'][task]={'counts':dict(counts),'by_condition':dict(by),'failure_reasons':dict(reasons),'unique_physical_resets':complete['unique_physical_resets'],'elapsed_s':complete['elapsed_s']}
    summary['total_cases']=len(accepted)+len(rejected);summary['initialization_pass']=len(accepted);summary['initialization_fail']=len(rejected)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'initialization_pass_ids.json').write_text(json.dumps(accepted,indent=2)+'\n')
    (root/'rejected.json').write_text(json.dumps(rejected,indent=2)+'\n')
    report=['# OOD 후보 원격 초기화 검사 결과','',f"전체 {summary['total_cases']}개: 초기화 검사 통과 {len(accepted)}개, 탈락 {len(rejected)}개.",'','이 검사는 정책 성공률 평가가 아니다. 통과 사례도 rollout_ready=false이며 최종 OOD 테스트셋으로 승인하지 않았다. 동일 base의 ID와 관측 노이즈 조건은 물리 상태가 같아 초기화 결과를 공유한다.','', '| 태스크 | 통과 | 탈락 | 실제로 검사한 고유 물리 상태 |','|---|---:|---:|---:|']
    for t,s in summary['tasks'].items():report.append(f"| {t} | {s['counts'].get('pass',0)} | {s['counts'].get('fail',0)} | {s['unique_physical_resets']} |")
    report+=['','## 방법과 한계','','- D3IL 고정 커밋의 실제 MuJoCo 장면을 생성하고 후보 위치·방향을 반영했다. 시작 손끝 변형은 관절 한계가 있는 수치 IK로 검사했다.','- 물체를 0.6초 동안 실제 물리 엔진으로 안정화했다. 로봇 관절은 이 단계에서 고정했으므로 컨트롤러 유지 성능이나 정책 실행 검증이 아니다.','- 원본 reset의 초기 물체-지지면 관통은 안정화 이전에만 허용했다. 최종 충돌, XY 이동 2 mm 초과, yaw 이동 0.02 rad 초과, 잔여 선속도 0.01 m/s·각속도 0.1 rad/s 초과, 지지 접촉 부재를 검사했다.','- 기존 손가락끼리의 자기접촉은 canonical reset에서 측정한 쌍만 예외 처리했다. 모델 자체의 collision mask/exclude는 그대로 사용했다.','- 정책 관측 노이즈를 실제 정책 입력으로 연결한 시험은 아직 아니다. 노이즈 생성 재현성과 분리 규칙만 검증했다.','- 작업공간의 운영상 한계, 각 물체에 대한 접근/파지 및 목표까지 경로, 실제 학습 데이터의 분포, controller/history 초기화 검증이 남아 있다.','- 고의로 블록을 겹치게 한 사례와 도달 불가능한 시작점을 넣은 음성 대조군에서 충돌/IK 오류가 검출됨을 확인했다.','- 원본 7,440개 후보는 수정하지 않았다. 탈락 사유를 별도 기록했다. 조건별 통과율이 달라 통과 ID만으로 균형 잡힌 최종 평가셋이 완성되지는 않는다.','','## 조건별 결과','','| 태스크 / 조건 | 통과 | 탈락 |','|---|---:|---:|']
    for t,s in summary['tasks'].items():
        for c,n in s['by_condition'].items():report.append(f"| {t} / {c} | {n.get('pass',0)} | {n.get('fail',0)} |")
    (root/'REPORT_KO.md').write_text('\n'.join(report)+'\n')
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--input',required=True);a=p.parse_args();print(json.dumps(summarize(a.results,a.input),indent=2))
