"""Inserting evaluation entry point using the shared execution loop."""

from configs.config import parse_config
from .evaluation import evaluate


def main():
    evaluate(parse_config('eval', task='inserting'))


if __name__ == '__main__':
    main()
