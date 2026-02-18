import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.train import train

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()

    train(
        level=args.level,
        num_episodes=args.episodes,
        device=args.device,
        log_interval=args.log_interval,
    )
