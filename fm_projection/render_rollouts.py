"""Replay saved evaluation rollouts in the D3IL simulator and record videos.

The policy is NOT re-run.  Each rollout's executed actions (``<ep>_act`` in
``rollouts.npz``) are fed back from the same initial context with exactly the
float64 accumulation the eval loop used (des_{k+1} = des_k + a_k), so the robot
motion is reproduced; MuJoCo is deterministic, and the replay error against the
saved hand / box states is printed and written to ``replay_check.json``.

Frame layout, one column per --eval-dir (e.g. baseline | --inject every):

    header  : label, outcome, step / sim time, desired-actual hand gap
    camera  : D3IL cage camera ``bp_cam`` rendered offscreen
    map     : top-down view in the camera's orientation (x down, y right):
              actual hand path (blue), commanded = desired hand path (orange),
              their current gap (red when > --track-d), boxes, targets, walls

    # inside the container (needs: apt-get install -y libosmesa6)
    python render_rollouts.py --task pushing \\
        --eval-dir ../fm_baseline/results/pushing_baseline/eval_pushing_test_<baseline> \\
        --eval-dir ../fm_baseline/results/pushing_baseline/eval_pushing_test_<every> \\
        --label baseline --label every --auto 3

    # no simulator / GL at all: map panel only, from the saved arrays
    python render_rollouts.py --task pushing --eval-dir ... --map-only --auto 3
"""
import argparse
import copy
import json
import os
import sys


def _early_gl():
    """MUJOCO_GL / PYOPENGL_PLATFORM must be fixed before any GL import."""
    gl = "osmesa"
    for i, a in enumerate(sys.argv):
        if a == "--gl" and i + 1 < len(sys.argv):
            gl = sys.argv[i + 1]
        elif a.startswith("--gl="):
            gl = a.split("=", 1)[1]
    os.environ.setdefault("MUJOCO_GL", gl)
    os.environ.setdefault("PYOPENGL_PLATFORM", gl)


_early_gl()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BASE = os.path.join(REPO, "fm_baseline")
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)

TASK = {
    # box half-size, target half-size (from the D3IL object files), default stride / fps
    "pushing": {"n_boxes": 2, "box_half": 0.03, "target_half": 0.05, "stride": 2, "fps": 15,
                "box_rgb": [(220, 40, 40), (40, 170, 40)]},
    "inserting": {"n_boxes": 3, "box_half": 0.025, "target_half": 0.025, "stride": 5, "fps": 20,
                  "box_rgb": [(220, 40, 40), (40, 170, 40), (40, 70, 220)]},
}
HAND = (30, 110, 230)
DES = (245, 140, 20)
BAD = (220, 30, 30)
GRID = (232, 232, 232)
INK = (40, 40, 40)
HEADER_H = 44
SEP = 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=sorted(TASK), required=True)
    p.add_argument("--eval-dir", action="append", required=True, help="eval_* folder; repeat to put runs side by side")
    p.add_argument("--label", action="append", default=None, help="panel title per --eval-dir (ASCII)")
    p.add_argument("--episode", action="append", default=None, help="episode key such as c003_t1; repeatable")
    p.add_argument("--auto", type=int, default=0, help="pick N episodes automatically (see pick_episodes)")
    p.add_argument("--stride", type=int, default=None, help="render every k env steps")
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--size", type=int, default=384, help="camera / map panel size in px (even)")
    p.add_argument("--track-d", type=float, default=0.10, help="gap [m] above which the des-hand line turns red")
    p.add_argument("--map-only", action="store_true", help="no simulator: draw the map panel from saved arrays")
    p.add_argument("--gl", choices=["osmesa", "egl", "glfw"], default="osmesa")
    p.add_argument("--d3il-root", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--raw-root", default="/root/d3il-work/raw/inserting")
    p.add_argument("--contexts-file", default=os.path.join(BASE, "contexts", "inserting_val.pkl"))
    p.add_argument("--out-dir", default=None, help="default: <first eval dir>/videos")
    args = p.parse_args()
    if args.size % 2:
        args.size += 1
    # setup_d3il() chdir()s into the D3IL repo, so freeze every path now
    for k in ("d3il_root", "data_dir", "raw_root", "contexts_file", "out_dir"):
        v = getattr(args, k)
        if v:
            setattr(args, k, os.path.abspath(v))
    return args


# --------------------------------------------------------------------------- #
# Saved evaluations
# --------------------------------------------------------------------------- #
def load_eval(d):
    d = os.path.abspath(d)
    with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
        res = json.load(f)
    z = np.load(os.path.join(d, "rollouts.npz"))
    roll = {k: z[k] for k in z.files}
    eps = sorted(k[:-4] for k in roll if k.endswith("_des"))
    return {"dir": d, "res": res, "roll": roll, "episodes": eps}


def ep_index(ep):
    ci, ti = ep.split("_t")
    return int(ci[1:]), int(ti)


def ep_success(ev, ep):
    ci, ti = ep_index(ep)
    return bool(ev["roll"]["successes"][ci, ti])


def check_same_contexts(evals, task):
    r0 = evals[0]["res"]
    for ev in evals[1:]:
        r = ev["res"]
        if task == "pushing" and r.get("contexts") != r0.get("contexts"):
            raise SystemExit(f"context set differs: {r0.get('contexts')} vs {r.get('contexts')} ({ev['dir']})")
        if task == "inserting" and r.get("context_files") != r0.get("context_files"):
            raise SystemExit(f"context files differ between {evals[0]['dir']} and {ev['dir']}")


def pick_episodes(evals, n):
    """With >1 runs: first those where run 0 failed and another run succeeded
    ('fixed'), then run 0 succeeded but another failed ('broken'), then all
    failed.  With one run: failures first, then successes."""
    common = sorted(set.intersection(*[set(e["episodes"]) for e in evals]))
    s = {ep: [ep_success(e, ep) for e in evals] for ep in common}
    if len(evals) > 1:
        groups = [("fixed", [ep for ep in common if not s[ep][0] and any(s[ep][1:])]),
                  ("broken", [ep for ep in common if s[ep][0] and not all(s[ep][1:])]),
                  ("all_fail", [ep for ep in common if not any(s[ep])]),
                  ("all_success", [ep for ep in common if all(s[ep])])]
    else:
        groups = [("fail", [ep for ep in common if not s[ep][0]]),
                  ("success", [ep for ep in common if s[ep][0]])]
    out = []
    for reason, eps in groups:
        for ep in eps:
            if len(out) < n:
                out.append((ep, reason))
    return out


def resolve_data_dir(arg, run_dir):
    if arg:
        return arg
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        saved = json.load(f).get("data_dir", "")
    if saved and os.path.isdir(saved):
        return saved
    return os.path.join(REPO, "data", os.path.basename(saved.rstrip("/")))


# --------------------------------------------------------------------------- #
# Simulator replay
# --------------------------------------------------------------------------- #
def quat_wxyz_to_mat(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def footprint(pos, quat, half):
    """XY corners of a box (half extents `half`) at pos with WXYZ quat."""
    R = quat_wxyz_to_mat(quat)
    ex, ey = R[:2, 0] * half[0], R[:2, 1] * half[1]
    c = np.asarray(pos[:2], dtype=np.float64)
    return np.stack([c + ex + ey, c - ex + ey, c - ex - ey, c + ex - ey])


class Sim:
    def __init__(self, args, ev0):
        from train import resolve_d3il
        self.task = args.task
        d3il_root = os.path.abspath(resolve_d3il(args.d3il_root))
        if args.task == "pushing":
            from eval_pushing import read_state, setup_d3il
            Env, self.quat2euler = setup_d3il(d3il_root)
            cfile = os.path.join(d3il_root, "environments/dataset/data/pushing", f"{ev0['res']['contexts']}_contexts.pkl")
            contexts = np.load(cfile, allow_pickle=True)
            self.get_ctx = lambda ci: contexts[ci]
        else:
            from eval_inserting import load_contexts, read_state, setup_d3il
            Env, self.quat2euler = setup_d3il(d3il_root)
            run_dir = os.path.dirname(ev0["dir"])
            cfg = {"data_dir": resolve_data_dir(args.data_dir, run_dir)}
            split = ev0["res"]["contexts"].split("_")[0]
            items = load_contexts(cfg, split, args.raw_root, self.quat2euler, 0, args.contexts_file)
            fmap = {f: ctx for f, _, ctx, _ in items}
            files = ev0["res"]["context_files"]
            self.get_ctx = lambda ci: fmap[files[ci]]
        self.read_state = read_state
        self.env = Env(render=False, n_substeps=35, max_steps_per_episode=10 ** 7)
        self.env.start()
        self.size = args.size
        self._gl_ok = False

    def statics(self):
        """Targets and (inserting) maze walls as XY polygons, after a reset."""
        env, polys = self.env, []
        names = ["target_box_1", "target_box_2"] if self.task == "pushing" else ["target_box1", "target_box2", "target_box3"]
        for i, n in enumerate(names):
            obj = getattr(env, n, None)
            if obj is not None:
                polys.append(("target", i, footprint(env.scene.get_obj_pos(obj), env.scene.get_obj_quat(obj), obj.size)))
        if self.task == "inserting":
            maze = getattr(sys.modules.get(type(env).__module__), "maze", None) or []
            for obj in maze:
                try:
                    polys.append(("wall", -1, footprint(env.scene.get_obj_pos(obj), env.scene.get_obj_quat(obj), obj.size)))
                except Exception:  # noqa: BLE001
                    pass
        return polys

    def camera(self):
        try:
            img = self.env.bp_cam.get_image(width=self.size, height=self.size, depth=False)
        except Exception as e:  # noqa: BLE001
            if not self._gl_ok:
                raise SystemExit(
                    f"offscreen rendering failed with MUJOCO_GL={os.environ.get('MUJOCO_GL')}: {e!r}\n"
                    "  osmesa: apt-get install -y libosmesa6  (CPU, works in any container)\n"
                    "  egl   : container needs -e NVIDIA_DRIVER_CAPABILITIES=all\n"
                    "  or run with --map-only")
            raise
        self._gl_ok = True
        return np.ascontiguousarray(img[..., :3]).astype(np.uint8)

    def replay(self, ctx, acts, need, record):
        env = self.env
        env.reset(random=False, context=copy.deepcopy(ctx))
        c, boxes = self.read_state(env, self.quat2euler)
        fixed_z = c[2:3].copy()
        des = c[:2].copy()                          # same start as the eval loop
        T = len(acts)
        out = {"des": np.zeros((T, 2)), "c": np.zeros((T, 2)), "boxes": [], "imgs": {},
               "des0": des.copy(), "c0": c[:2].copy(),
               "boxes0": np.concatenate([np.r_[b[0][:2], b[2]] for b in boxes]),
               "statics": self.statics()}
        if record and -1 in need:
            out["imgs"][-1] = self.camera()
        for k in range(T):
            des = des + acts[k].astype(np.float64)  # identical float64 accumulation
            env.step(np.concatenate([des, fixed_z, [0.0, 1.0, 0.0, 0.0]]))
            c, boxes = self.read_state(env, self.quat2euler)
            out["des"][k], out["c"][k] = des, c[:2]
            out["boxes"].append(np.concatenate([np.r_[b[0][:2], b[2]] for b in boxes]))
            if record and k in need:
                out["imgs"][k] = self.camera()
        out["boxes"] = np.asarray(out["boxes"])
        return out


def from_saved(ev, ep, statics):
    r = ev["roll"]
    des, c, boxes = r[ep + "_des"].astype(np.float64), r[ep + "_c"][:, :2].astype(np.float64), r[ep + "_boxes"].astype(np.float64)
    a0 = r[ep + "_act"][0].astype(np.float64)
    return {"des": des, "c": c, "boxes": boxes, "imgs": {},
            "des0": des[0] - a0, "c0": c[0], "boxes0": boxes[0], "statics": statics}


def map_only_statics(args, ev0):
    """Without the simulator only the Inserting targets are known (manifest)."""
    if args.task != "inserting":
        return []
    try:
        ddir = resolve_data_dir(args.data_dir, os.path.dirname(ev0["dir"]))
        with open(os.path.join(ddir, "manifest.json"), encoding="utf-8") as f:
            tps = json.load(f)["fixed_target_positions"]
    except Exception:  # noqa: BLE001
        return []
    h = TASK["inserting"]["target_half"]
    return [("target", i, footprint(tp, [1, 0, 0, 0], [h, h])) for i, tp in enumerate(tps)]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
class MapView:
    """World XY -> pixels in the bp_cam orientation: y to the right, x downward."""

    def __init__(self, size, pts, pad=14):
        lo, hi = pts.min(0) - 0.04, pts.max(0) + 0.04
        span = float(max(hi - lo))
        self.cx, self.cy = (lo + hi) / 2
        self.x0, self.y0 = self.cx - span / 2, self.cy - span / 2
        self.s = (size - 2 * pad) / span
        self.pad, self.size, self.span = pad, size, span

    def px(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        u = self.pad + (xy[..., 1] - self.y0) * self.s
        v = self.pad + (xy[..., 0] - self.x0) * self.s
        return np.stack([u, v], -1).round().astype(np.int32)

    def base(self, statics, task):
        img = np.full((self.size, self.size, 3), 255, np.uint8)
        step = 0.1
        for g in np.arange(np.floor(self.x0 / step) * step, self.x0 + self.span + 1e-9, step):
            p = self.px([[g, self.y0], [g, self.y0 + self.span]])
            cv2.line(img, tuple(p[0]), tuple(p[1]), GRID, 1)
        for g in np.arange(np.floor(self.y0 / step) * step, self.y0 + self.span + 1e-9, step):
            p = self.px([[self.x0, g], [self.x0 + self.span, g]])
            cv2.line(img, tuple(p[0]), tuple(p[1]), GRID, 1)
        cols = TASK[task]["box_rgb"]
        for kind, i, poly in statics:
            pts = self.px(poly).reshape(-1, 1, 2)
            if kind == "wall":
                cv2.fillPoly(img, [pts], (150, 150, 150))
            else:
                col = tuple(int(255 - 0.35 * (255 - v)) for v in cols[i % len(cols)])
                cv2.fillPoly(img, [pts], col)
                cv2.polylines(img, [pts], True, cols[i % len(cols)], 1, cv2.LINE_AA)
        cv2.putText(img, "top view  x down / y right  grid 0.1 m", (6, self.size - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1, cv2.LINE_AA)
        return img


def draw_map(view, base, tr, k, task, track_d):
    img = base.copy()
    half = TASK[task]["box_half"]
    cols = TASK[task]["box_rgb"]
    nb = TASK[task]["n_boxes"]
    kk = max(k, -1)
    des = np.vstack([tr["des0"][None], tr["des"][:kk + 1]])
    hand = np.vstack([tr["c0"][None], tr["c"][:kk + 1]])
    boxes = np.vstack([tr["boxes0"][None], tr["boxes"][:kk + 1]])
    for b in range(nb):                                   # faint box trails
        p = view.px(boxes[:, 3 * b:3 * b + 2]).reshape(-1, 1, 2)
        faint = tuple(int(255 - 0.3 * (255 - v)) for v in cols[b])
        cv2.polylines(img, [p], False, faint, 1, cv2.LINE_AA)
    cv2.polylines(img, [view.px(des).reshape(-1, 1, 2)], False, DES, 1, cv2.LINE_AA)
    cv2.polylines(img, [view.px(hand).reshape(-1, 1, 2)], False, HAND, 2, cv2.LINE_AA)
    cur = boxes[-1]
    for b in range(nb):                                   # current boxes
        x, y, yaw = cur[3 * b:3 * b + 3]
        q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        pts = view.px(footprint([x, y], q, [half, half])).reshape(-1, 1, 2)
        cv2.fillPoly(img, [pts], cols[b])
    d, h = des[-1], hand[-1]
    gap = float(np.linalg.norm(d - h))
    pd, ph = view.px(d), view.px(h)
    cv2.line(img, tuple(ph), tuple(pd), BAD if gap > track_d else (120, 120, 120), 1, cv2.LINE_AA)
    cv2.circle(img, tuple(ph), 5, HAND, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(pd), 4, DES, 2, cv2.LINE_AA)
    y0 = 16
    for txt, col in [("actual hand", HAND), ("desired hand (command)", DES)]:
        cv2.line(img, (8, y0 - 4), (26, y0 - 4), col, 2)
        cv2.putText(img, txt, (32, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK, 1, cv2.LINE_AA)
        y0 += 16
    return img, gap


def header(width, label, ok, k, T, gap, track_d):
    img = np.full((HEADER_H, width, 3), 250, np.uint8)
    tag, col = ("SUCCESS", (30, 140, 50)) if ok else ("FAIL", BAD)
    cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, INK, 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(img, tag, (width - tw - 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    step = max(k + 1, 0)
    info = f"step {step}/{T}  t={step * 0.035:5.2f}s  |des-hand|={gap:.3f} m"
    cv2.putText(img, info, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BAD if gap > track_d else INK, 1, cv2.LINE_AA)
    return img


def compose(cols):
    h = max(c.shape[0] for c in cols)
    cols = [np.vstack([c, np.full((h - c.shape[0], c.shape[1], 3), 255, np.uint8)]) for c in cols]
    out = []
    for i, c in enumerate(cols):
        if i:
            out.append(np.full((h, SEP, 3), 200, np.uint8))
        out.append(c)
    frame = np.hstack(out)
    H, W = frame.shape[:2]                                 # even dims for x264
    return np.pad(frame, ((0, H % 2), (0, W % 2), (0, 0)), constant_values=255)


class Writer:
    def __init__(self, path, fps):
        self.path, self.fps, self.w, self.kind = path, fps, None, None

    def add(self, rgb):
        if self.w is None:
            try:
                import imageio.v2 as iio
                self.w = iio.get_writer(self.path, fps=self.fps, codec="libx264", quality=8, macro_block_size=1)
                self.kind = "imageio"
            except Exception:  # noqa: BLE001
                h, w = rgb.shape[:2]
                self.w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                self.kind = "cv2"
        if self.kind == "imageio":
            self.w.append_data(rgb)
        else:
            self.w.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.w is not None:
            self.w.close() if self.kind == "imageio" else self.w.release()


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    cfg = TASK[args.task]
    stride = args.stride or cfg["stride"]
    fps = args.fps or cfg["fps"]
    evals = [load_eval(d) for d in args.eval_dir]
    labels = args.label or [os.path.basename(e["dir"]) for e in evals]
    if len(labels) != len(evals):
        raise SystemExit("give one --label per --eval-dir")
    check_same_contexts(evals, args.task)

    if args.episode:
        todo = [(ep, "manual") for ep in args.episode]
    else:
        todo = pick_episodes(evals, args.auto or 1)
    for ep, _ in todo:
        for e in evals:
            if ep not in e["episodes"]:
                raise SystemExit(f"{ep} not in {e['dir']}")

    out_dir = os.path.abspath(args.out_dir or os.path.join(evals[0]["dir"], "videos"))
    os.makedirs(out_dir, exist_ok=True)
    sim = None if args.map_only else Sim(args, evals[0])
    statics_saved = map_only_statics(args, evals[0]) if args.map_only else None
    report = []

    for ep, reason in todo:
        ci, _ = ep_index(ep)
        Ts = [len(e["roll"][ep + "_act"]) for e in evals]
        Tmax = max(Ts)
        grid = ([] if args.map_only else [-1]) + list(range(stride - 1, Tmax, stride))
        if grid[-1] != Tmax - 1:
            grid.append(Tmax - 1)
        traces, checks = [], []
        for e, T in zip(evals, Ts):
            acts = e["roll"][ep + "_act"]
            if sim is None:
                tr = from_saved(e, ep, statics_saved)
            else:
                need = {min(k, T - 1) for k in grid}
                tr = sim.replay(sim.get_ctx(ci), acts, need, record=True)
                r = e["roll"]
                chk = {"des": float(np.abs(tr["des"].astype(np.float32) - r[ep + "_des"]).max()),
                       "hand": float(np.abs(tr["c"] - r[ep + "_c"][:, :2]).max()),
                       "boxes": float(np.abs(tr["boxes"] - r[ep + "_boxes"]).max())}
                checks.append(chk)
            traces.append(tr)

        pts = [tr["des"] for tr in traces] + [tr["c"] for tr in traces]
        pts += [tr["boxes"][:, 3 * b:3 * b + 2] for tr in traces for b in range(cfg["n_boxes"])]
        pts += [poly for _, _, poly in traces[0]["statics"]]
        view = MapView(args.size, np.vstack(pts))
        bases = [view.base(tr["statics"], args.task) for tr in traces]

        name = f"{args.task}_{ep}_" + "_vs_".join(l.replace(" ", "-") for l in labels)
        path = os.path.join(out_dir, name + ".mp4")
        wr = Writer(path, fps)
        last = None
        for k in grid:
            cols = []
            for e, tr, T, lab, base in zip(evals, traces, Ts, labels, bases):
                kk = min(k, T - 1)
                m, gap = draw_map(view, base, tr, kk, args.task, args.track_d)
                parts = [header(args.size, lab, ep_success(e, ep), kk, T, gap, args.track_d)]
                if sim is not None:
                    parts.append(tr["imgs"][kk])
                parts.append(m)
                cols.append(np.vstack(parts))
            last = compose(cols)
            wr.add(last)
        wr.close()
        cv2.imwrite(os.path.join(out_dir, name + "_last.png"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
        item = {"episode": ep, "reason": reason, "steps": Ts, "success": [ep_success(e, ep) for e in evals],
                "video": path, "frames": len(grid), "writer": wr.kind}
        if checks:
            item["replay_max_abs_err"] = checks
        report.append(item)
        msg = f"{ep} [{reason}] steps {Ts} success {item['success']} -> {path} ({len(grid)} frames, {wr.kind})"
        if checks:
            msg += "  replay err " + " | ".join(f"des {c['des']:.1e} hand {c['hand']:.1e} boxes {c['boxes']:.1e}" for c in checks)
        print(msg, flush=True)

    with open(os.path.join(out_dir, "replay_check.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if any("replay_max_abs_err" in r and max(c["hand"] for c in r["replay_max_abs_err"]) > 1e-4 for r in report):
        print("WARNING: replay diverged from the saved rollout (> 0.1 mm); the video is not an exact reproduction.")


if __name__ == "__main__":
    main()
