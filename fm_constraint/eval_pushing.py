"""Pushing evaluation entry point; policy and observations come from the baseline."""

from .config import parse_config
from .evaluation import evaluate


def main():
    evaluate(parse_config('eval', task='pushing'))


if __name__ == '__main__':
    main()
