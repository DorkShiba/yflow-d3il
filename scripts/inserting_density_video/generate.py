"""Build the two A_10 density videos shown on slides 3 and 4."""

import argparse
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from density import PALETTE

ROBOT_SIZE = (720, 504)
DEFAULT_FONT = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
PAIRS = [("I08", 0), ("I00", 0), ("I08", 1)]


def compose(work, destination, font, *, outcomes):
    keys = ["I08-seed0", "I08-seed1"] if outcomes else ["I00-seed0", "I08-seed0"]
    names = (
        ["② PF2MP  ·  seed 0", "② PF2MP  ·  seed 1"]
        if outcomes
        else ["기본 정책", "② PF2MP"]
    )
    title = (
        "② PF2MP의 성공 경로와 실패 경로"
        if outcomes
        else "학습 시연의 위치 밀도와 로봇팔의 삽입 경로"
    )
    case = "A_10  /  배치 13" + ("" if outcomes else "  /  seed 0")
    subject = "두 실행" if outcomes else "두 정책"
    fonts = {
        n: ImageFont.truetype(str(font), n) for n in (19, 21, 22, 23, 24, 26, 32, 34)
    }
    paper = (249, 250, 247)
    ink = "#173338"
    faint = "#456369"
    accent = "#006E64"
    high = []
    reached = []
    successes = []
    for key in keys:
        with np.load(work / (key + ".npz")) as run:
            high.append(run["high"])
            reached.append(run["reached"])
            successes.append(bool(run["success"]))
    if outcomes and successes != [True, False]:
        raise ValueError(
            "The two replays did not reproduce the reported success/failure pair"
        )
    lengths = [len(counts) - 1 for counts in high]
    end = min(lengths)
    frame_counts = [(steps + 3) // 4 + 1 for steps in lengths]
    last = [None, None]
    with ExitStack() as stack:

        def pipe(*args, **kwargs):
            return stack.enter_context(subprocess.Popen(*args, **kwargs))

        readers = [
            pipe(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-i",
                    str(work / (key + ".mp4")),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-",
                ],
                stdout=subprocess.PIPE,
            )
            for key in keys
        ]
        encoder = pipe(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "1440x788",
                "-r",
                "25",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            stdin=subprocess.PIPE,
        )
        for i in range(max(frame_counts) + 35):
            t = min(4 * i, max(lengths))
            im = Image.new("RGB", (1440, 788), paper)
            dr = ImageDraw.Draw(im)

            def label(x, y, s, sz, col=ink, anchor="lm", draw=dr):
                draw.text((x, y), s, font=fonts[sz], fill=col, anchor=anchor)

            label(26, 36, title, 34)
            label(1413, 36, case + "  ·  CPU 재실행  ·  3.5배속", 22, faint, "rm")
            dr.rectangle((0, 70, 1439, 121), fill="#18383D")
            for j, p in enumerate(readers):
                off = 720 * j
                cur = min(t, lengths[j])
                k = min(cur, end)
                if i < frame_counts[j]:
                    raw = p.stdout.read(720 * 504 * 3)
                    if len(raw) != 720 * 504 * 3:
                        raise ValueError(f"Truncated camera clip {keys[j]}, frame {i}")
                    last[j] = Image.frombytes("RGB", ROBOT_SIZE, raw)
                im.paste(last[j], (off, 122))
                label(off + 26, 94, names[j], 26, "white")
                status = (
                    ("성공" if successes[j] else "시간 종료")
                    if cur == lengths[j]
                    else "진행 중"
                )
                label(
                    off + 695, 94, f"{cur:,} 스텝   ·   {status}", 22, "#D9ECE7", "rm"
                )
                n = int(reached[j][cur])
                share = f"{100 * high[j][k] / k:.1f}%" if k else "—"
                label(off + 26, 653, "상위 25% 영역 체류율", 23, faint)
                label(off + 263, 652, share, 32, accent)
                label(off + 695, 652, f"목표 도달  {n} / 3", 24, ink, "rm")
            dr.line([(720, 70), (720, 677)], fill="#C4D0CF", width=2)
            dr.line([(26, 681), (1414, 681)], fill="#BBCBC8", width=1)
            label(26, 714, "학습 위치 밀도", 23)
            for x, co, txt in zip(
                [209, 388, 600, 819],
                PALETTE,
                ["하위 5%", "5~50백분위", "50~75백분위", "상위 25%"],
            ):
                dr.rectangle((x, 701, x + 30, 724), fill=co, outline="#899D9C")
                label(x + 42, 713, txt, 21)
            label(1413, 714, "검은 선  최근 손 목표 경로", 21, faint, "rm")
            label(
                26,
                758,
                f"바닥색은 시연 속 손 목표 XY의 KDE 밀도. 체류율은 {subject}의 공통 구간(최대 {end:,}스텝)까지 누적.",
                21,
                faint,
            )
            label(1413, 758, "목표 도달 기준  XY 1 cm 이내", 19, faint, "rm")
            encoder.stdin.write(im.tobytes())
        encoder.stdin.close()
        for reader in readers:
            if reader.stdout.read(1) or reader.wait():
                raise ValueError("Unexpected camera clip length or decoder error")
        if encoder.wait():
            raise RuntimeError("Failed to encode comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="evaluation checkout with model and dataset (presentation revision: 8d0c7aba)",
    )
    parser.add_argument(
        "--d3il", required=True, type=Path, help="D3IL simulator checkout"
    )
    parser.add_argument(
        "--runs",
        type=Path,
        help="existing context directory containing I00-seed0, I08-seed0 and I08-seed1; omit to evaluate on CPU",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="destination for both MP4s"
    )
    parser.add_argument(
        "--font", type=Path, default=DEFAULT_FONT, help="Korean TTF/TTC/OTF font"
    )
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error("Select a Korean font file with --font")
    source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        work = Path(stack.enter_context(TemporaryDirectory(dir=args.output))).resolve()
        runs = args.runs
        if runs is None:
            run_root = source / "fm_constraint/runs"
            run_root.mkdir(parents=True, exist_ok=True)
            runs = Path(
                stack.enter_context(
                    TemporaryDirectory(dir=run_root, prefix="density-video-")
                )
            )
        for method, seed in PAIRS:
            key = f"{method}-seed{seed}"
            run_dir = runs / key
            cmd = [
                sys.executable,
                str(Path(__file__).with_name("render.py").resolve()),
                str(source),
                str(args.d3il.resolve()),
                str(run_dir.resolve()),
                str(work / key),
                method,
                str(seed),
            ]
            if args.runs is None:
                cmd.append("--evaluate")
            subprocess.run(cmd, check=True, cwd=source)
        movies = []
        for outcomes, suffix in [(False, "학습밀도"), (True, "성공실패-밀도")]:
            movie = work / f"A10-배치13-PF2MP-{suffix}.mp4"
            compose(work, movie, args.font, outcomes=outcomes)
            movies.append(movie)
        for movie in movies:
            target = args.output / movie.name
            os.replace(movie, target)
            print(target)


if __name__ == "__main__":
    main()
