"""학습 코드를 복제하지 않고 baseline trainer에 검증한 설정을 전달한다."""

from datetime import datetime, timezone
from pathlib import Path
import sys

from . import checkpoint_reads, import_baseline
from configs.config import parse_config


def main():
    cfg = parse_config('train')
    trainer = import_baseline('train')
    if not cfg.values['train']['resume']:
        # 입력 run에 쓰지 않도록 이번 학습 전용 부모 폴더 아래에 새 run을 만든다.
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
        # 중단되더라도 생성된 checkpoint 옆에 resolved 설정을 함께 남긴다.
        for child in output.iterdir():
            if child.is_dir() and (child / 'config.json').is_file():
                cfg.save(child)


if __name__ == '__main__':
    main()
