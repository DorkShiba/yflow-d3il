"""Launch a matched baseline/RTC pair, then compare their saved results."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["pushing", "inserting"], required=True)
    p.add_argument("--output-root")
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args()
    if any(x.split("=")[0] in ("--method", "--output", "--task") for x in extra):
        p.error("method/output/task are controlled by the comparison runner")
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.output_root).resolve() if args.output_root else repo / "codex/rtc/runs" / (
        args.task + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if root.exists():
        raise FileExistsError(root)
    commands = [[sys.executable, "-m", "fm_rtc.evaluate", "--task", args.task,
                 "--method", method, "--output", str(root / method), *extra]
                for method in ("baseline", "rtc")]
    commands.append([sys.executable, "-m", "fm_rtc.compare", "--baseline", str(root / "baseline/results.json"),
                     "--rtc", str(root / "rtc/results.json"), "--output", str(root / "comparison.json")])
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
