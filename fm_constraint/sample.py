"""Reuse the baseline's offline generation checks without writing to its runs."""

from datetime import datetime, timezone
from pathlib import Path
import sys

from . import checkpoint_reads, import_baseline
from configs.config import parse_config


def main():
    cfg = parse_config('sample')
    sampler = import_baseline('sample')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    cfg.save(Path(cfg.values['paths']['output_root']) / f'sample_{cfg.task}_{stamp}')
    old_argv = sys.argv
    try:
        sys.argv = ['fm_baseline.sample', *cfg.baseline_argv('sample')]
        with checkpoint_reads():
            sampler.main()
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    main()
