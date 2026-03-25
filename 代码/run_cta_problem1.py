from __future__ import annotations

import argparse
from pathlib import Path

from src.cta_problem1.runner import StrategyConfig, run_problem_1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CTA problem 1 backtest on domestic futures main contracts.")
    parser.add_argument(
        "--data-dir",
        default="main&smain_market_data_01min",
        help="Directory that stores the 1-minute main contract feather files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cta_problem_1",
        help="Directory for reports, metrics, and charts.",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol list for smoke tests.",
    )
    parser.add_argument(
        "--signal-freq",
        default="240min",
        help="Signal bar frequency, for example 240min, 120min, 60min, or 30min.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()] or None
    result = run_problem_1(
        data_dir=project_root / args.data_dir,
        output_dir=project_root / args.output_dir,
        config=StrategyConfig(signal_freq=args.signal_freq),
        symbols=symbols,
    )
    print(f"output_dir: {result['output_dir']}")
    print(f"report_path: {result['report_path']}")
    print(f"symbol_metrics_path: {result['symbol_metrics_path']}")
    print(f"portfolio_metrics_path: {result['portfolio_metrics_path']}")


if __name__ == "__main__":
    main()
