"""Run the unchanged baseline trainer with externally supplied settings."""

from datetime import datetime, timezone
from pathlib import Path
import sys

from . import checkpoint_reads, import_baseline
from .config import parse_config


def main():
    cfg = parse_config('train')
    trainer = import_baseline('train')
    if not cfg.values['train']['resume']:
        # Give this invocation its own parent; the baseline creates its run below it.
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        cfg.values['paths']['output_root'] = str(Path(cfg.values['paths']['output_root']) / f'train_{stamp}')
    output = Path(cfg.values['train']['resume'] or cfg.values['paths']['output_root'])
    cfg.save(output)
    old_argv = sys.argv
    try:
        sys.argv = ['fm_baseline.train', *cfg.baseline_argv('train')]
        with checkpoint_reads():
            trainer.main()
    finally:
        sys.argv = old_argv
        # Include the resolved YAML beside each produced checkpoint, even after interruption.
        for child in output.iterdir():
            if child.is_dir() and (child / 'config.json').is_file():
                cfg.save(child)


if __name__ == '__main__':
    main()
