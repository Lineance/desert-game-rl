import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _main() -> None:
    from src.utils.validator import validator_main

    validator_main()


if __name__ == "__main__":
    _main()
