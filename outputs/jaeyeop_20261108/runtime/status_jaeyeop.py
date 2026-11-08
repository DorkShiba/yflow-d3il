import collections, datetime, json, pathlib, subprocess
repo = pathlib.Path('/root/d3il-work/yflow-d3il')
root = pathlib.Path('/root/d3il-work/jaeyeop_20261108')
print('UTC', datetime.datetime.now(datetime.timezone.utc).isoformat())
for f in ('started_utc.txt', 'finished_utc.txt', 'exitcode'):
    p = root/f
    if p.exists(): print(f, p.read_text().strip())
units = []
for p in (repo/'experiments/id_v1').glob('*/*/ctx_*/result.json'):
    u=json.loads(p.read_text())
    if u['context_index'] in (17,18,19): units.append(u)
print('completed units', len(units), '/ 192; rollouts', sum(len(u['episodes']) for u in units), '/ 576')
for task in ('pushing', 'inserting'):
    us = [u for u in units if u['task']==task]
    print(task, 'units', len(us), 'wall_s_sum', round(sum(u['wall_s'] for u in us),2), 'failures', dict(collections.Counter(e['failure'] for u in us for e in u['episodes'])))
    for u in sorted(us,key=lambda u:u['finished_utc'])[-6:]:
        print(' ', u['combo'], u['context_index'], round(u['wall_s'],2), [(e['success'],e['steps'],e['failure']) for e in u['episodes']])
for p in sorted(root.glob('ctx*.log')):
    lines=p.read_text(errors='replace').splitlines()
    print(p.name, '\n'+'\n'.join(lines[-5:]))
for p in sorted(root.glob('*.exitcode')): print(p.name,p.read_text().strip())
