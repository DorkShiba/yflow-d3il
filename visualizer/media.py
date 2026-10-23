"""Finite rollout import and shared, physical-time based export renderer."""
import re
import numpy as np
from PIL import Image, ImageDraw

COLORS = ['#b64b3b', '#36836b', '#8062a4']

def read_rollouts(path, dt):
    if not 0 < dt <= 1:
        raise ValueError('Step duration must be in (0, 1] seconds')
    result = []
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(data.files):
            match = re.fullmatch(r'(c(\d+)_t(\d+))_c', key)
            if not match:
                continue
            base, ci, ti = match.group(1), int(match.group(2)), int(match.group(3))
            arrays = {name: np.asarray(data[base + '_' + name]) for name in ['c', 'des', 'boxes']}
            if base + '_joints' in data:
                arrays['joints'] = np.asarray(data[base + '_joints'])
            for name in ['act', 'boxes_xyz']:
                if base + '_' + name in data:
                    arrays[name] = np.asarray(data[base + '_' + name])
            n = len(arrays['c'])
            if not 1 <= n <= 20000:
                raise ValueError('Invalid rollout length')
            for name, arr in arrays.items():
                if arr.ndim != 2 or len(arr) != n or not np.isfinite(arr).all():
                    raise ValueError('Nonfinite or inconsistent rollout arrays')
            if arrays['c'].shape[1] < 2 or arrays['des'].shape[1] != 2 or arrays['boxes'].shape[1] not in (6, 9):
                raise ValueError('Unsupported state dimensions')
            if 'joints' in arrays and arrays['joints'].shape != (n, 7):
                raise ValueError('Panda joint record must have shape (steps, 7)')
            if 'act' in arrays and arrays['act'].shape != (n, 2):
                raise ValueError('Actions must have shape (steps, 2)')
            if 'boxes_xyz' in arrays and arrays['boxes_xyz'].shape != arrays['boxes'].shape:
                raise ValueError('XYZ block log dimensions mismatch')
            success = bool(data['successes'][ci, ti]) if 'successes' in data else None
            extra = {}
            if base + '_goals' in data:
                goals=np.asarray(data[base+'_goals'],dtype=float)
                threshold=float(data[base+'_goal_threshold'])
                rule=str(data[base+'_goal_rule'])
                if goals.shape != (arrays['boxes'].shape[1]//3,3) or not np.isfinite(goals).all() or not 0 < threshold < 1 or rule not in ('matched','any_permutation'):
                    raise ValueError('Invalid goal metadata')
                if rule == 'any_permutation' and len(goals)!=2: raise ValueError('Permutation requires two goals')
                extra['goal_spec']=dict(goals=goals.tolist(),threshold=threshold,rule=rule,description='색상 교차 허용' if rule=='any_permutation' else '블록별 지정 목표',source='Recorded simulator goal positions')
            if base + '_plan_starts' in data:
                starts = np.asarray(data[base + '_plan_starts'])
                if starts.shape != (n,): raise ValueError('Invalid plan boundaries')
                extra['plan_starts'] = starts.astype(bool).tolist()
            result.append(dict(**extra, key=base, steps=n, dt=dt, success=success,
                               **{k: v.tolist() for k, v in arrays.items()}))
    if not result:
        raise ValueError('No D3IL rollouts found')
    return result

def bounds(rolls):
    points = []
    for r in rolls:
        points.extend(np.asarray(r['c'])[:, :2]); points.extend(r['des'])
        b = np.asarray(r['boxes'])
        for i in range(0, b.shape[1], 3):
            points.extend(b[:, i:i+2])
        if r.get('goal_spec'):
            spec=r['goal_spec'];radius=spec['threshold']
            for goal in spec['goals']:
                points.extend([np.asarray(goal[:2])-radius,np.asarray(goal[:2])+radius])
    a = np.asarray(points)
    lo, hi = a.min(0), a.max(0)
    center = (lo + hi) / 2
    half = max(float((hi-lo).max()) / 2 + .04, .1)
    return [*(center-half), *(center+half)]

def frame(r, t, label, box, size=(640, 480), robot_dir=None):
    im = Image.new('RGB', size, '#ffffff'); draw = ImageDraw.Draw(im)
    w, h = size
    # Saved states are post-step: first sample occurs at dt, not t=0.
    k = min(max(int(t / r['dt'] + 1e-9) - 1, 0), r['steps']-1)
    if robot_dir is not None:
        with Image.open(robot_dir / f'{k:06d}.jpg') as src:
            src.thumbnail((w, h-60)); im.paste(src, ((w-src.width)//2, 45))
    else:
        x0,y0,x1,y1 = box
        scale = min((w-60)/(x1-x0), (h-100)/(y1-y0))
        def xy(p):
            return (w/2+(p[0]-(x0+x1)/2)*scale, (h+40)/2-(p[1]-(y0+y1)/2)*scale)
        if r.get('goal_spec'):
            spec=r['goal_spec'];radius=spec['threshold']*scale
            for index,goal in enumerate(spec['goals']):
                x,y=xy(goal);color=COLORS[index]
                draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=2)
                draw.text((x+8,y-12),f'G{index+1}',fill=color)
        paths = [(r['c'], '#25669b'), (r['des'], '#697680')]
        b = np.asarray(r['boxes'])
        paths += [(b[:,j:j+2], COLORS[j//3]) for j in range(0,b.shape[1],3)]
        for pts,color in paths:
            line = [xy(p) for p in pts[:k+1]]
            if len(line)>1: draw.line(line, fill=color, width=2)
            x,y=line[-1]; draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
        draw.text((15,h-24), 'hand: blue / target hand: gray / blocks: colored', fill='#58636e')
    draw.text((15,12), label[:80], fill='#20252b')
    state = 'finished' if t >= r['steps']*r['dt'] else 'playing'
    draw.text((15,28), f"{r['key']} | step {k+1}/{r['steps']} | {t:.2f}s | {state} | success={r['success']}", fill='#58636e')
    return im
