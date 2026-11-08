from pathlib import Path
import json, math, hashlib, collections
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
units=[json.loads(p.read_text()) for p in sorted((ROOT/'id_v1').glob('*/*/ctx_*/result.json'))]
assert len(units)==192 and sum(len(u['episodes']) for u in units)==576
assert len({u['code_fingerprint'] for u in units})==1
assert len({(u['task'],u['number'],u['context_index']) for u in units})==192
assert all(sorted(e['sampling_seed'] for e in u['episodes'])==[0,1,2] for u in units)
assert {u['member'] for u in units}=={'jaeyeop'}
noise={}
for u in units:
 for e in u['episodes']:
  key=(u['task'],u['context_index'],e['sampling_seed'])
  assert key not in noise or noise[key]==e['noise_seed']
  noise[key]=e['noise_seed']

def avg(v):
 v=[float(x) for x in v if x is not None and math.isfinite(x)]
 return sum(v)/len(v) if v else float('nan')

G={}
for task in ('pushing','inserting'):
 for n in range(32):
  us=[u for u in units if u['task']==task and u['number']==n]
  es=[e for u in us for e in u['episodes']]
  ents=[]
  for u in us:
   counts=collections.Counter(e['mode'] for e in u['episodes'] if e['success'])
   total=sum(counts.values())
   if total: ents.append(-sum((x/total)*math.log(x/total) for x in counts.values())/math.log(4 if task=='pushing' else 6))
  G[task,n]={'id':('P' if task=='pushing' else 'I')+f'{n:02d}','success':sum(e['success'] for e in es),
   'violation':100*sum(e['constraint_violation_steps'] for e in es)/sum(e['constraint_evaluated_steps'] for e in es),
   'change':avg([e['action_change_m'] for e in es])*1000,'jerk':avg([e['command_jerk_m_s3'] for e in es]),
   'entropy':avg(ents),'failure':sum(e['failure'] not in (None,'max_steps') for e in es),
   'time':avg([u['wall_s'] for u in us])}

def fmt(x,d=3): return f'{x:.{d}g}'
doc=Document();sec=doc.sections[0]
sec.page_width=Inches(8.27);sec.page_height=Inches(11.69)
sec.top_margin=Inches(.65);sec.bottom_margin=Inches(.65);sec.left_margin=Inches(.7);sec.right_margin=Inches(.7)
sec.header_distance=Inches(.25);sec.footer_distance=Inches(.25)
for name in ('Normal','Title','Subtitle','Heading 1','Heading 2','Caption'):
 s=doc.styles[name];s.font.name='Arial Unicode MS';s.font.color.rgb=RGBColor(0,0,0)
 fonts=s.element.get_or_add_rPr().rFonts
 for key in list(fonts.attrib): del fonts.attrib[key]
 for key in ('ascii','hAnsi','eastAsia','cs'): fonts.set(qn('w:'+key),'Arial Unicode MS')
 s.paragraph_format.space_after=Pt(7)
doc.styles['Normal'].font.size=Pt(10.5);doc.styles['Normal'].paragraph_format.line_spacing=Pt(14.5)
doc.styles['Normal'].paragraph_format.space_after=Pt(5)
doc.styles['Title'].font.size=Pt(25);doc.styles['Heading 1'].font.size=Pt(17);doc.styles['Heading 2'].font.size=Pt(12)
doc.styles['Caption'].font.size=Pt(9);doc.styles['Caption'].font.italic=False
doc.core_properties.title='Y Flow 재엽 담당 ID 실험 최종 보고서'
doc.core_properties.author='정재엽'
doc.core_properties.subject='Pushing과 Inserting의 5개 추론 시점 방법 32조합 평가'
footer=sec.footer.paragraphs[0];footer.alignment=2
r=footer.add_run();r.font.size=Pt(9)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');r._r.addnext(field)

def p(text,style=None):return doc.add_paragraph(text,style)
def h(text):doc.add_heading(text,level=1)
def h2(text):doc.add_heading(text,level=2)
def page():doc.add_page_break()
def table(headers,rows,widths=None,size=9.2):
 t=doc.add_table(rows=1, cols=len(headers));t.autofit=False
 if widths:
  for col,w in zip(t.columns,widths):col.width=Inches(w)
 for c,txt in zip(t.rows[0].cells,headers):c.text=str(txt)
 trPr=t.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');trPr.append(repeat)
 for row in rows:
  for c,txt in zip(t.add_row().cells,row):c.text=str(txt)
 for i,row in enumerate(t.rows):
  pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
  for j,c in enumerate(row.cells):
   if widths:c.width=Inches(widths[j])
   c.vertical_alignment=1
   for para in c.paragraphs:
    para.paragraph_format.space_after=Pt(2);para.paragraph_format.space_before=Pt(2);para.paragraph_format.line_spacing=Pt(12)
    for r in para.runs:r.font.size=Pt(size);r.bold=(i==0)
   if i==0:
    shading=OxmlElement('w:shd');shading.set(qn('w:fill'),'EDEFF2');c._tc.get_or_add_tcPr().append(shading)
 tail=p('');tail.paragraph_format.space_after=Pt(0);tail.paragraph_format.line_spacing=Pt(3)
 return t

def results(task,nums):
 rows=[]
 for n,label in nums:
  r=G[task,n]
  rows.append([r['id']+' '+label,f"{r['success']}/9",fmt(r['violation']),f"{r['change']:.2f}",f"{r['jerk']:.1f}"])
 table(['조합','성공','위반율 %','행동 변화 mm','Jerk m/s³'],rows,[2.7,.6,.9,1.1,1.1])

doc.add_heading('Y Flow 재엽 담당\nID 실험 최종 보고서',0)
p('Pushing과 Inserting의 추론 시점 방법 조합 평가', 'Subtitle')
p('정재엽  |  실험 및 보고 기준일 2026년 11월 8일')
h('1 연구 결과 요약')
p('본 실험은 학습된 Flow Matching 정책을 고정한 상태에서 Projection, PF2MP, Filter, RTC, LiPo의 적용 조합을 평가했다. 정재엽 담당 초기 상태인 context 17–19에서 태스크별 32조합을 수행했으며, 총 576회 rollout을 모두 저장했다. 이 보고서는 해당 3개 context의 결과를 독립적으로 해석한 최종 기록이다.')
p('Pushing에서는 RTC와 LiPo의 병용이 9/9 성공과 jerk 약 30% 감소를 함께 보였다. Inserting에서는 Projection과 PF2MP의 병용이 9/9 성공했으며, PF2MP와 RTC의 병용은 8/9 성공과 측정 제약 위반율 0%를 보였다. 다섯 방법을 모두 켠 조합은 두 태스크 모두 7/9 성공해, 방법을 추가할수록 성능이 높아지는 관계는 관찰되지 않았다.')
p('가장 큰 부정적 상호작용은 Inserting의 PF2MP와 LiPo 병용에서 나타났다. 각 단독 방법은 8/9 성공했지만 병용은 2/9에 그쳤다. 계획 및 궤적 로그는 작은 행동 보정 이후 접촉과 재계획을 거치며 경로가 갈라지고 일부 실행이 정체되는 양상을 보였다. 구체적인 접촉 원인은 추가 검증이 필요하다.')
table(['항목','수행 범위 및 결과'],[
 ['초기 상태와 반복','태스크별 context 17, 18, 19 × noise seed 0, 1, 2'],
 ['평가 규모','두 태스크 × 32조합 × 3 context × 3 seed = 576 rollout'],
 ['완료 상태','192/192 단위 완료  |  576/576 rollout 저장'],
 ['실행 시간','2026년 11월 8일 13:41:04–16:57:10 KST  |  3시간 16분 6초'],
 ['장비와 실행','RTX 3090 24GB 한 장에서 context별 세 프로세스 실행'],
 ['해석 범위','조합당 9회 점추정  |  팀 전체 30 context의 성능을 대표하지 않음']],[1.35,5.05])
p('성공 1회의 차이는 11.1%p에 해당한다. 표본의 독립적인 초기 상태는 3개이며, 9회를 서로 독립인 9개 환경 표본으로 취급하지 않는다. 본 보고서에서는 유의성이나 일반화 우열을 주장하지 않는다.')

page();h('2 실험 설계와 평가 방법')
h2('비교 방법과 조합 식별자')
table(['방법','적용 방식','비트'],[
 ['Projection','Predict의 각 Euler 갱신 직후 행동 계획을 제약 집합으로 공동 투영','16'],
 ['PF2MP','학습 데이터 KDE potential로 미래 상태의 목표 XY 채널에 guidance 적용','8'],
 ['Filter','실행 직전 최신 관측으로 행동 하나를 투영하며 수정 시 재계획','4'],
 ['RTC','직전 계획의 미실행 행동을 참조해 계획 간 연속성 유도','2'],
 ['LiPo','계획 끝에서 jerk QP 적용  |  누적 위치 보정 축당 최대 0.3mm','1']],[1.05,4.95,.4])
p('조합 번호는 활성 방법의 비트 합이다. P와 I는 각각 Pushing과 Inserting을 뜻한다. 예를 들어 P03은 RTC와 LiPo, I24는 Projection과 PF2MP이며 00은 baseline, 31은 전체 ON이다. 각 조합은 동일한 context와 seed를 공유한다.')
h2('고정 조건과 실행 관리')
p('태스크별 EMA checkpoint 1개를 사용했다. Euler 적분 10회, horizon 16, 한 번에 8개 행동 실행, 제어 간격 0.035초를 고정했다. 최대 길이는 Pushing 400 step, Inserting 2,000 step이다. PF2MP weight 0.01, bandwidth 0.025m, alpha −1, RTC β 5, LiPo 보정 한도 0.3mm를 변경하지 않았다. Inserting은 validation 초기 상태이므로 test 일반화 결과로 해석하지 않는다.')
p('동일 context의 모든 조합을 같은 장비에서 실행했다. 모든 결과의 코드 fingerprint와 git commit이 일치했으며, 각 평가 키는 중복 없이 존재했다. 첫 계획의 noise seed가 조합 간 일치함을 결과 파일에서 확인했다. 24-step 동작 확인은 본실험에 포함하지 않았다.')
h2('지표와 집계 규칙')
table(['지표','정의와 집계'],[
 ['성공률','D3IL 공식 판정에 따른 성공 수 / 9  |  방법 실패도 분모에 포함'],
 ['제약 위반율','workspace·가속도·object envelope 중 하나라도 위반한 실행 step / 전체 평가 step. 모든 rollout의 step을 합산'],
 ['행동 변화','평균 ‖aₜ − aₜ₋₁‖  |  각 rollout 평균 후 context와 seed에 동일 가중치'],
 ['명령 jerk','평균 ‖aₜ − 2aₜ₋₁ + aₜ₋₂‖ / Δt³  |  실행 명령 기준이며 실제 관절 jerk가 아님'],
 ['Entropy','성공 mode의 정규화 entropy를 context별 계산 후 평균. 성공 없는 context는 제외'],
 ['시간','context 하나에서 seed 3회를 완료하는 평균 벽시계 시간. 동시 실행 부하 포함']],[1.1,5.3],size=9)
p('측정 제약 위반율 0%는 이 지표에서 위반을 관측하지 않았다는 뜻이다. 충돌 없음이나 물리적 안전성을 별도로 입증하지 않는다. 신뢰구간은 산출하지 않았으며 결과는 모두 점추정이다.')

page();h('3 태스크별 주요 결과')
h2('Pushing에서는 RTC와 LiPo 병용이 성공과 jerk를 함께 개선했다')
results('pushing',[(0,'Baseline'),(1,'LiPo'),(2,'RTC'),(3,'RTC + LiPo'),(16,'Projection'),(9,'PF2MP + LiPo'),(31,'전체 ON')])
r0=G['pushing',0];r3=G['pushing',3]
p(f"RTC와 LiPo의 병용은 baseline 대비 성공을 8/9에서 9/9로 높이고 jerk를 {r0['jerk']:.1f}에서 {r3['jerk']:.1f}로 {(1-r3['jerk']/r0['jerk'])*100:.1f}% 줄였다. RTC 단독도 9/9 성공했으며, LiPo 추가 후 jerk가 더 낮아졌다. 다만 성공률에서 RTC 단독보다 추가 이득은 관찰되지 않았다. Projection 단독도 9/9 성공해 단일 우승 방법을 확정할 수는 없다.")
p('전체 ON은 jerk 30.5로 더 낮았지만 성공은 7/9였다. 매끄러운 명령과 과제 성공의 개선 방향이 항상 일치하지 않았다. 전체 ON의 낮은 jerk만으로 우수성을 판단해서는 안 된다.')
h2('Inserting에서는 조합별 성공과 제약 위반 사이의 차이가 컸다')
results('inserting',[(0,'Baseline'),(1,'LiPo'),(8,'PF2MP'),(10,'PF2MP + RTC'),(24,'Projection + PF2MP'),(25,'Projection + PF2MP + LiPo'),(9,'PF2MP + LiPo'),(31,'전체 ON')])
p('Projection과 PF2MP의 병용은 9/9 성공했지만 제약 위반율이 0.0478% 남았다. PF2MP와 RTC는 8/9 성공하면서 위반율 0%와 jerk 60.4를 보였다. Projection·PF2MP·LiPo는 8/9 성공, 위반율 0%, jerk 59.4를 기록해 다른 균형점의 후보가 된다. 이 세 조합의 우열은 성공·위반·명령 변동 중 어떤 지표를 우선하는지에 따라 달라진다.')
p('LiPo 단독은 성공과 jerk를 개선했지만 행동 변화는 1.98mm에서 2.12mm로 늘었다. 따라서 단독 LiPo가 모든 평활도 지표를 개선했다고 해석하지 않는다. 전체 ON도 성공률 최고 조합보다 낮았다.')

page();h('4 PF2MP와 LiPo 병용의 실패 분석')
p('Inserting에서 PF2MP와 LiPo는 각각 8/9 성공했지만 병용은 2/9였다. 같은 context·seed로 대응시키면 PF2MP 단독 성공 8건 중 7건이 병용에서 실패했고, 단독 실패 1건은 병용에서 성공했다. 성공률 차이는 −66.7%p이며, 2/9는 7회의 수행 실패를 포함한 값이다.')
paired=[]
for c in (17,18,19):
 for s in (0,1,2):
  row=[str(c),str(s)]
  for n in (1,8,9):
   u=next(u for u in units if (u['task'],u['number'],u['context_index'])==('inserting',n,c));e=next(e for e in u['episodes'] if e['sampling_seed']==s)
   row.append(('성공' if e['success'] else '실패')+f"  {e['steps']}")
  paired.append(row)
table(['Context','Seed','LiPo I01','PF2MP I08','병용 I09'],paired,[.75,.55,1.7,1.7,1.7])
h2('로그에서 직접 확인한 사실')
p('병용 I09의 1,988개 계획에서 계획 및 LiPo 상태는 모두 정상이고, 보정 한도도 지켜졌다. 각 계획의 LiPo jerk 목적함수는 적용 후 모두 증가하지 않았다. Inserting 9개 대응 rollout 모두에서 첫 LiPo 적용 전 행동 계획은 PF2MP 단독과 완전히 같았고, 첫 noise seed도 일치했다. Solver 실패나 다른 초기 계획은 I09의 성능 하락을 설명하지 못한다.')
p('Context 17·seed 0에서는 작은 보정 후 경로 차이가 커졌다. PF2MP 단독과 병용의 블록 위치 차이는 53 step에서 1cm를, 64 step에서 5cm를 넘었다. PF2MP 단독은 1,003 step에 성공했지만 병용은 녹색 블록만 처리한 뒤 2,000 step까지 성공하지 못했다.')
h2('다른 조합의 결과로 확인한 설명의 범위')
table(['I09에 추가한 방법','성공','위반율 %'],[['없음','2/9','1.06'],['Filter','2/9','0'],['RTC','6/9','0'],['Projection','8/9','0']],[3.4,1.3,1.7])
p('Filter로 측정 위반을 제거해도 성공은 2/9에 머물렀다. 따라서 가속도 등 측정 위반만으로 실패를 설명하기 어렵다. RTC나 Projection 추가 시 성공이 회복된 결과는 계획·제어의 상호작용 가설과 부합하지만, 회복 원인까지 분리한 실험은 아니다.')

page();h('5 궤적에서 확인한 실패 양상')
doc.add_picture(str(ROOT/'pf2mp_lipo_analysis/trajectories.png'),width=Inches(6.8))
doc.paragraphs[-1].paragraph_format.line_spacing=1.0
doc.paragraphs[-1].paragraph_format.keep_with_next=True
p('그림 1  같은 초기 상태와 seed에서 PF2MP 단독과 병용의 물체 경로. 회색은 실측 손, 색 선은 블록 경로, 점선 원은 목표 영역의 XY 투영이다. 오른쪽 그래프는 PF2MP 단독을 점선, 병용을 실선으로 표시한다. 성공 기준은 XY 거리가 아닌 3차원 거리 1cm이다.','Caption')
h2('물체를 멀리 남기는 실패와 목표 근처의 정체')
p('위 행의 context 17·seed 0에서 병용은 빨간 블록을 초기 위치에 남기고 파란 블록도 목표로 가져오지 못했다. 최종 XY 목표 거리는 각각 약 444mm와 500mm였다. 마지막 200 step 동안 세 블록의 최대 순변위는 0.001mm 미만인 반면, 손과 명령의 이동은 계속됐다. 태스크 진전 없이 제어가 반복된 사례다.')
p('병용 실패 7건 중 5건은 최소 한 블록을 목표에서 0.18m 이상 떨어뜨린 채 종료했다. 나머지 2건인 context 18·seed 2와 context 19·seed 1은 최종 XY 위치가 목표 근처였지만 공식 성공 판정을 받지 못했다. 전자는 마지막 200 step에 블록 순변위가 0이었고, 후자의 최대 순변위는 약 0.86mm였다.')
p('목표 근처의 두 사례는 XY 거리만으로 판정을 재현할 수 없다. 현재 전체 계획 로그에는 블록 XY가 저장돼 있으나 블록 Z와 접촉력은 없어, 높이·접촉 상태 중 무엇이 성공을 막았는지 확정하지 않는다. 또한 n_boxes는 D3IL의 과거 목표 방문 이력에 기반하므로 최종 동시 정렬 개수와 동일하게 취급하지 않는다.')
p('Pushing에서는 context 18·seed 1 한 건이 PF2MP 단독 성공에서 병용 실패로 바뀌었다. Jerk는 65.4에서 48.8로 낮아졌지만 평균 목표 거리는 약 31.0mm에서 91.6mm로 늘었다. 단 한 건의 변화이므로 Inserting과 같은 실패 기전을 일반화하지 않는다.')

page();h('6 계획 평활화와 실행 결과의 차이')
doc.add_picture(str(ROOT/'pf2mp_lipo_analysis/divergence.png'),width=Inches(6.8))
doc.paragraphs[-1].paragraph_format.line_spacing=1.0
doc.paragraphs[-1].paragraph_format.keep_with_next=True
p('그림 2  Context 17·seed 0의 경로 분기와 명령 지표. 위 왼쪽은 PF2MP 단독과 병용의 위치 차이이며, 위 오른쪽은 각 실행 내 목표 손 위치와 실측 손 XY의 차이다. 아래 왼쪽 jerk는 40-step 이동평균, 아래 오른쪽은 세 블록의 누적 이동 거리 합이다.','Caption')
p('이 사례에서 LiPo는 각 계획의 jerk 목적함수 합계를 19.3% 낮췄지만, 실행 전체의 명령 jerk는 72.4에서 144.9m/s³로 증가했다. 8-step chunk 전환 시점과 직후의 평균 jerk는 130.7에서 362.4m/s³로 커졌다. 병용의 최종 목표 손과 실측 손의 XY 차이는 93.7mm였고, 후반부에는 블록 이동 없이 높은 jerk와 큰 추종 오차가 반복됐다.')
p('LiPo의 한도는 매 계획에서 원래 계획 대비 누적 위치의 축별 차이를 제한한다. 서로 다른 관측에서 재계획하는 전체 rollout 사이의 경로 차이를 제한하지는 않는다. 초기 보정이 작더라도 접촉 상태와 다음 관측이 달라지면 후속 FM 계획도 달라질 수 있다. 위 로그는 이러한 폐루프 민감성을 직접 보여 주지만, 단일 접촉 사건의 원인까지 식별하지는 못한다.')
p('현재 코드에서 PF2MP는 Predict 중 미래 목표 상태 XY 채널을 유도하고, LiPo는 Predict 후 행동을 보정한다. 단순히 동일한 행동 채널에서 반대 힘을 더하는 구조는 아니다. 저장된 계획의 누적 행동 경로에 대해 PF potential을 전후 재계산했을 때에도 에너지가 일관되게 악화되지 않았다. 따라서 LiPo가 PF2MP 유도를 항상 지운다는 설명은 지지되지 않는다.')

page();h('7 결론과 후속 검증')
h2('이번 담당 범위에서 도출한 결론')
p('Pushing에서는 RTC와 LiPo가 성공률을 유지 또는 개선하면서 jerk를 낮추는 실용적인 후보였다. Inserting에서는 Projection과 PF2MP가 성공률 우선 후보였고, PF2MP와 RTC 또는 Projection·PF2MP·LiPo는 관측된 제약 위반과 jerk까지 고려할 후보였다. 전체 ON이 가장 우수하다는 근거는 없었다.')
p('Inserting의 PF2MP와 LiPo 병용 저하는 동일 초기 상태와 seed에서도 나타났다. Solver 실패와 보정 한도 초과는 관찰되지 않았다. 경로의 초기 분기, 후반 물체 정체, 일부 사례의 chunk 경계 jerk 증가는 작은 행동 보정이 접촉·관측·재계획을 거치며 증폭된다는 가설을 뒷받침한다. 이 가설은 로그에 가장 잘 맞는 설명이며, 확정된 접촉 수준의 원인은 아니다.')
h2('증거와 해석의 한계')
p('본 결과는 ID 조건에서 3개 context만 평가한 결과다. 각 조합의 9회 중 3회는 동일 context의 noise 반복이며, 32조합을 살펴본 후 좋은 후보를 선택했으므로 선택 편향이 있다. 통계적 유의성, 전체 D3IL 성능, OOD 강건성, 물리적 안전성은 주장하지 않는다. Entropy도 context당 성공 최대 3회의 작은 표본에서 계산된 값이다.')
p('4건의 solver_failure는 Inserting I16·I18·I20·I22의 동일 context 17·seed 0에서 발생했다. 실패를 분모에 포함했고 결과를 대체하지 않았다. 서로 다른 네 환경에서 발생한 독립 사건으로 해석하지 않는다. 시간 지표는 GPU를 공유한 병렬 실행의 벽시계 시간이므로 조합별 순수 계산 비용이나 단독 실행 속도의 순위를 뜻하지 않는다.')
h2('후속 실험 제안')
p('첫째, 별도 진단 실험에서 동일 실패 context·seed와 코드를 고정하고 LiPo OFF 및 보정 한도 0.1·0.2·0.3mm를 비교한다. 성공률뿐 아니라 첫 경로 분기 시점, 블록 진행 정체, chunk 경계 jerk를 함께 기록하면 보정 크기와 실패의 관계를 확인할 수 있다. 진단 결과는 본실험 성능표와 분리한다.')
p('둘째, 블록 Z, 접촉 대상과 힘, 매 step 목표 거리와 phase 전환을 저장해 목표 근처 실패와 접촉 경로 이탈을 구분한다. 방법 파라미터를 선택하거나 조정하려면 본 평가용 초기 상태와 분리된 calibration context를 사용한다. 이번 보고서 작성 과정에서는 추가 rollout이나 파라미터 변경을 수행하지 않았다.')
h2('재현성과 자료 위치')
p('실험 코드 commit 4d33329fe332a6c7687d5bb9e25d36997fb61344, fingerprint 0167b01b63c192f42f6ca1af9813d6d64b923cab17ed6642d4e8c5faffabb669. D3IL commit은 1d9c71850d4fc1477cee17b557c8b97d70b13071이다.')
p('근거 자료는 같은 결과 폴더의 id_v1 단위별 result.json 및 trajectories.npz, report_jaeyeop.md, pf2mp_lipo_analysis의 metrics.json·paired_comparisons.json·potential_diagnostics.json이다. 원격 서버의 원본 전체 계획 로그를 분석했으며 source_manifest.json에 분석 대상 24개 로그의 SHA256을 보존했다. 부록 수치는 단위별 결과를 재집계했다.')

for task,label in [('pushing','Pushing'),('inserting','Inserting')]:
 page();h('부록 '+label+' 전체 32조합')
 p('모든 행은 context 17–19와 seed 0–2의 총 9회 결과다. 조합 번호 = Projection 16 + PF2MP 8 + Filter 4 + RTC 2 + LiPo 1. 위반율은 step 합산 기준이며 시간은 context당 seed 3회 평균이다.')
 rows=[]
 for n in range(32):
  r=G[task,n];rows.append([r['id'],f"{r['success']}/9",fmt(r['violation']),f"{r['change']:.2f}",f"{r['jerk']:.1f}",fmt(r['entropy']),f"{r['failure']}/9",f"{r['time']:.0f}"])
 table(['조합','성공','위반 %','Δa mm','Jerk','Entropy','방법 실패','시간 s'],rows,[.6,.6,.85,.8,.8,.8,.9,.75],size=9)
 p('Jerk 단위는 m/s³이며 평균 행동 변화 Δa는 mm다. 방법 실패는 solver 등 방법 자체의 실패이며 성공하지 못한 모든 rollout 수와는 다르다. 성공률의 비교 기준은 같은 부록의 00 baseline이다.','Caption')
 if task=='pushing':p('Pushing context 원본 ID  |  17 → 32, 18 → 36, 19 → 38. 공식 test 초기 상태 중 담당 범위다.')
 else:p('Inserting validation 원본 ID  |  17 → env_078_04.npz, 18 → env_083_02.npz, 19 → env_090_03.npz.')

path=OUT/'YFlow_재엽_ID실험_최종보고서_20261108.docx'
for el in doc.styles.element.xpath('.//w:pBdr'):
 el.getparent().remove(el)
for el in doc.element.xpath('.//w:pBdr'):
 el.getparent().remove(el)
for s in doc.styles:
 if s.type==1:
  s.font.color.rgb=RGBColor(0,0,0)
for para in doc.paragraphs:
 for run in para.runs:
  run.font.color.rgb=RGBColor(0,0,0)
for el in doc.element.xpath('.//w:docGrid'):
 el.getparent().remove(el)
doc.save(path)
summary={'units':192,'rollouts':576,'contexts':[17,18,19],'seeds':[0,1,2],'git_commit':units[0]['git_commit'],'fingerprint':units[0]['code_fingerprint'],'metrics':[{'task':t,**r} for (t,n),r in G.items()]}
(OUT/'report_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
manifest=[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((ROOT/'id_v1').glob('*/*/ctx_*/result.json'))]
(OUT/'result_manifest.json').write_text(json.dumps(manifest,indent=2))
print(path)
