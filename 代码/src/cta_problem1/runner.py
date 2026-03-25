from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

UNIVERSE = [
    "SP",
    "P",
    "BU",
    "CU",
    "V",
    "FG",
    "CF",
    "SC",
    "AU",
    "AG",
    "MA",
    "PG",
    "RM",
    "RU",
    "TA",
    "EB",
    "SR",
    "FU",
    "I",
    "SN",
    "OI",
    "ZN",
    "Y",
    "SA",
    "M",
    "RB",
    "EG",
    "PP",
    "AL",
    "NI",
    "HC",
]

FEE_RATE = 0.0002
TRADE_COLUMNS = [
    "symbol",
    "direction",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "gross_return",
    "net_return",
    "holding_hours",
]


@dataclass(frozen=True)
class StrategyConfig:
    signal_freq: str = "240min"
    fast_ema: int = 12
    slow_ema: int = 36
    entry_lookback: int = 16
    exit_lookback: int = 8
    er_lookback: int = 8
    er_threshold: float = 0.15


def run_problem_1(
    data_dir: Path,
    output_dir: Path,
    config: StrategyConfig | None = None,
    symbols: list[str] | None = None,
) -> dict[str, Path]:
    config = config or StrategyConfig()
    symbols = symbols or UNIVERSE
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    symbol_metrics: list[dict[str, float | int | str]] = []
    all_trades: list[pd.DataFrame] = []
    daily_equity_frames: list[pd.Series] = []
    signal_bars_by_symbol: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        minute_df = load_symbol_data(data_dir=data_dir, symbol=symbol)
        signal_bars = build_signal_bars(minute_df=minute_df, config=config)
        signal_bars_by_symbol[symbol] = signal_bars
        target_signal = generate_target_signal(signal_bars=signal_bars, config=config)
        result = backtest_symbol(minute_df=minute_df, target_signal=target_signal, symbol=symbol)

        daily_equity = result["daily_equity"].rename(symbol)
        daily_equity_frames.append(daily_equity)
        all_trades.append(result["trades"])
        symbol_metrics.append(build_symbol_metrics(symbol=symbol, result=result))

    metrics_df = pd.DataFrame(symbol_metrics).sort_values("sharpe", ascending=False).reset_index(drop=True)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else empty_trades_frame()

    daily_equity_df = pd.concat(daily_equity_frames, axis=1).sort_index()
    daily_equity_df = daily_equity_df.ffill().fillna(1.0)
    portfolio_equity = daily_equity_df.mean(axis=1).rename("portfolio_equity")
    portfolio_metrics = compute_performance_metrics(portfolio_equity, trades_df)
    period_metrics = build_period_metrics(portfolio_equity=portfolio_equity, trades_df=trades_df)

    metrics_df.to_csv(output_dir / "symbol_metrics.csv", index=False)
    trades_df.to_csv(output_dir / "all_trades.csv", index=False)
    daily_equity_df.to_csv(output_dir / "symbol_daily_equity.csv", index=True)
    portfolio_equity.to_csv(output_dir / "portfolio_daily_equity.csv", index=True)
    pd.DataFrame([portfolio_metrics]).to_csv(output_dir / "portfolio_metrics.csv", index=False)
    pd.DataFrame(period_metrics).to_csv(output_dir / "portfolio_metrics_by_period.csv", index=False)

    plot_symbol_equities(daily_equity_df, charts_dir / "symbol_equity_curves.png")
    plot_portfolio_equity(portfolio_equity, charts_dir / "portfolio_equity.png")
    representative_paths = plot_representative_trade_charts(
        metrics_df=metrics_df,
        trades_df=trades_df,
        signal_bars_by_symbol=signal_bars_by_symbol,
        output_dir=charts_dir,
    )

    report_path = output_dir / "report.md"
    write_report(
        report_path=report_path,
        config=config,
        metrics_df=metrics_df,
        portfolio_metrics=portfolio_metrics,
        period_metrics=period_metrics,
        representative_paths=representative_paths,
    )
    return {
        "output_dir": output_dir,
        "report_path": report_path,
        "symbol_metrics_path": output_dir / "symbol_metrics.csv",
        "portfolio_metrics_path": output_dir / "portfolio_metrics.csv",
    }


def load_symbol_data(data_dir: Path, symbol: str) -> pd.DataFrame:
    file_path = data_dir / f"{symbol}_main.feather"
    df = pd.read_feather(file_path)
    timestamp = pd.to_datetime(df["tradeDate"] + " " + df["barTime"], format="%Y-%m-%d %H:%M")
    minute_df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "symbol": symbol,
            "hfq_open": df["hfq_openPrice"].astype(float),
            "hfq_high": df["hfq_highPrice"].astype(float),
            "hfq_low": df["hfq_lowPrice"].astype(float),
            "hfq_close": df["hfq_closePrice"].astype(float),
            "hfq_twap": df["hfq_twap"].astype(float),
            "volume": df["turnoverVol"].astype(float),
        }
    )
    minute_df = minute_df.dropna(subset=["timestamp", "hfq_twap", "hfq_close"])
    minute_df = minute_df.loc[minute_df["hfq_twap"] > 0].sort_values("timestamp").drop_duplicates("timestamp")
    return minute_df.set_index("timestamp")


def resample_hfq_bars(minute_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    bars = minute_df.resample(rule, label="right", closed="right").agg(
        {
            "hfq_open": "first",
            "hfq_high": "max",
            "hfq_low": "min",
            "hfq_close": "last",
            "hfq_twap": "mean",
            "volume": "sum",
        }
    )
    return bars.dropna(subset=["hfq_close"]).copy()


def build_signal_bars(minute_df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    signal_bars = resample_hfq_bars(minute_df, config.signal_freq)
    signal_bars["ema_fast"] = signal_bars["hfq_close"].ewm(span=config.fast_ema, adjust=False).mean()
    signal_bars["ema_slow"] = signal_bars["hfq_close"].ewm(span=config.slow_ema, adjust=False).mean()
    signal_bars["entry_high"] = signal_bars["hfq_close"].rolling(config.entry_lookback).max().shift(1)
    signal_bars["entry_low"] = signal_bars["hfq_close"].rolling(config.entry_lookback).min().shift(1)
    signal_bars["exit_high"] = signal_bars["hfq_close"].rolling(config.exit_lookback).max().shift(1)
    signal_bars["exit_low"] = signal_bars["hfq_close"].rolling(config.exit_lookback).min().shift(1)
    signal_bars["efficiency_ratio"] = efficiency_ratio(signal_bars["hfq_close"], config.er_lookback)
    return signal_bars


def efficiency_ratio(close: pd.Series, lookback: int) -> pd.Series:
    change = close.diff(lookback).abs()
    travel = close.diff().abs().rolling(lookback).sum()
    return (change / travel.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)


def generate_target_signal(signal_bars: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    position = 0
    targets: list[int] = []

    for _, row in signal_bars.iterrows():
        if row[["entry_high", "entry_low", "exit_high", "exit_low", "efficiency_ratio"]].isna().any():
            targets.append(position)
            continue

        long_entry = (
            row["hfq_close"] > row["entry_high"]
            and row["ema_fast"] > row["ema_slow"]
            and row["efficiency_ratio"] >= config.er_threshold
        )
        short_entry = (
            row["hfq_close"] < row["entry_low"]
            and row["ema_fast"] < row["ema_slow"]
            and row["efficiency_ratio"] >= config.er_threshold
        )
        long_exit = row["hfq_close"] < row["exit_low"] or row["ema_fast"] < row["ema_slow"]
        short_exit = row["hfq_close"] > row["exit_high"] or row["ema_fast"] > row["ema_slow"]

        if position == 0:
            if long_entry:
                position = 1
            elif short_entry:
                position = -1
        elif position > 0:
            if short_entry:
                position = -1
            elif long_exit:
                position = 0
        else:
            if long_entry:
                position = 1
            elif short_exit:
                position = 0

        targets.append(position)

    return pd.Series(targets, index=signal_bars.index, name="target_position")


def backtest_symbol(minute_df: pd.DataFrame, target_signal: pd.Series, symbol: str) -> dict[str, pd.DataFrame | pd.Series]:
    fill_updates = pd.Series(np.nan, index=minute_df.index, dtype=float)
    signal_index = target_signal.index.values
    fill_locations = minute_df.index.searchsorted(signal_index, side="right")
    valid_mask = fill_locations < len(minute_df.index)
    if valid_mask.any():
        fill_timestamps = minute_df.index[fill_locations[valid_mask]]
        updates = pd.Series(target_signal.values[valid_mask], index=fill_timestamps, dtype=float).groupby(level=0).last()
        fill_updates.loc[updates.index] = updates.values

    position_after_trade = fill_updates.ffill().fillna(0.0)
    position_before_bar = position_after_trade.shift(1).fillna(0.0)
    trade_delta = position_after_trade.diff().abs().fillna(position_after_trade.abs())

    price_returns = minute_df["hfq_twap"].pct_change().fillna(0.0)
    net_returns = position_before_bar * price_returns - trade_delta * FEE_RATE
    equity = 1.0 + net_returns.cumsum()
    daily_equity = equity.groupby(equity.index.normalize()).last()

    trades = reconstruct_trades(
        symbol=symbol,
        position_after_trade=position_after_trade,
        fill_prices=minute_df["hfq_twap"],
    )
    return {
        "minute_equity": equity.rename(symbol),
        "daily_equity": daily_equity.rename(symbol),
        "trades": trades,
    }


def reconstruct_trades(symbol: str, position_after_trade: pd.Series, fill_prices: pd.Series) -> pd.DataFrame:
    updates = position_after_trade[position_after_trade.ne(position_after_trade.shift()).fillna(position_after_trade.ne(0.0))]

    records: list[dict[str, float | str]] = []
    current_position = 0
    entry_time: pd.Timestamp | None = None
    entry_price: float | None = None

    for timestamp, new_position_value in updates.items():
        new_position = int(new_position_value)
        fill_price = float(fill_prices.loc[timestamp])

        if current_position != 0 and entry_time is not None and entry_price is not None and (
            new_position == 0 or int(math.copysign(1, new_position)) != current_position
        ):
            gross_return = current_position * (fill_price / entry_price - 1.0)
            net_return = gross_return - 2.0 * FEE_RATE
            holding_hours = (timestamp - entry_time).total_seconds() / 3600.0
            records.append(
                {
                    "symbol": symbol,
                    "direction": "long" if current_position > 0 else "short",
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "entry_price": entry_price,
                    "exit_price": fill_price,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "holding_hours": holding_hours,
                }
            )
            entry_time = None
            entry_price = None

        if new_position != 0 and new_position != current_position:
            current_position = 1 if new_position > 0 else -1
            entry_time = timestamp
            entry_price = fill_price
        else:
            current_position = new_position

    if not records:
        return empty_trades_frame()
    return pd.DataFrame(records, columns=TRADE_COLUMNS)


def build_symbol_metrics(symbol: str, result: dict[str, pd.DataFrame | pd.Series]) -> dict[str, float | int | str]:
    daily_equity = result["daily_equity"]
    trades = result["trades"]
    metrics = compute_performance_metrics(daily_equity, trades)
    metrics["symbol"] = symbol
    ordered = {"symbol": metrics.pop("symbol")}
    ordered.update(metrics)
    return ordered


def compute_performance_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict[str, float | int]:
    equity = equity.dropna()
    daily_returns = equity.diff().fillna(0.0)
    daily_std = float(daily_returns.std(ddof=0))
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1.0 / 252.0)
    total_return = float(equity.iloc[-1] - 1.0)
    annualized_return = total_return / years
    sharpe = 0.0 if daily_std == 0.0 else float(daily_returns.mean() / daily_std * math.sqrt(252.0))
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    max_drawdown = float(drawdown.min())
    calmar = 0.0 if max_drawdown == 0.0 else float(annualized_return / abs(max_drawdown))

    if trades.empty:
        win_rate = 0.0
        win_loss_ratio = 0.0
        avg_holding_hours = 0.0
        avg_return_per_trade = 0.0
        trade_count = 0
    else:
        winners = trades.loc[trades["net_return"] > 0.0, "net_return"]
        losers = trades.loc[trades["net_return"] < 0.0, "net_return"]
        win_rate = float((trades["net_return"] > 0.0).mean())
        if winners.empty or losers.empty:
            win_loss_ratio = float("inf") if not winners.empty else 0.0
        else:
            win_loss_ratio = float(winners.mean() / abs(losers.mean()))
        avg_holding_hours = float(trades["holding_hours"].mean())
        avg_return_per_trade = float(trades["net_return"].mean())
        trade_count = int(len(trades))

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "win_loss_ratio": win_loss_ratio,
        "number_of_transactions": trade_count,
        "average_holding_hours": avg_holding_hours,
        "average_profit_loss_per_transaction": avg_return_per_trade,
    }


def empty_trades_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_COLUMNS)


def build_period_metrics(portfolio_equity: pd.Series, trades_df: pd.DataFrame) -> list[dict[str, float | int | str]]:
    periods = [
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
    ]
    output: list[dict[str, float | int | str]] = []
    exit_times = pd.to_datetime(trades_df["exit_time"]) if not trades_df.empty else pd.Series(dtype="datetime64[ns]")
    for label, start_date, end_date in periods:
        period_equity = portfolio_equity.loc[start_date:end_date]
        period_trades = trades_df.loc[(exit_times >= pd.Timestamp(start_date)) & (exit_times <= pd.Timestamp(end_date))].copy()
        metrics = compute_performance_metrics(period_equity, period_trades)
        metrics["period"] = label
        output.append(metrics)
    return output


def plot_symbol_equities(daily_equity_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    for symbol in daily_equity_df.columns:
        ax.plot(daily_equity_df.index, daily_equity_df[symbol], linewidth=1.0, alpha=0.7, label=symbol)
    ax.set_title("Per-Symbol Daily Equity Curves")
    ax.set_ylabel("Net Value")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_portfolio_equity(portfolio_equity: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(portfolio_equity.index, portfolio_equity, color="#1f6feb", linewidth=2.0)
    ax.set_title("Equal-Weighted Portfolio Daily Equity")
    ax.set_ylabel("Net Value")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_representative_trade_charts(
    metrics_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    signal_bars_by_symbol: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    if trades_df.empty:
        return []

    chosen_symbols: list[str] = []
    eligible = metrics_df.loc[metrics_df["number_of_transactions"] > 0, "symbol"].tolist()
    if not eligible:
        return []

    chosen_symbols.append(eligible[0])
    chosen_symbols.append(metrics_df.loc[metrics_df["number_of_transactions"] > 0, "symbol"].iloc[-1])
    chosen_symbols.append(metrics_df.sort_values("number_of_transactions", ascending=False)["symbol"].iloc[0])

    unique_symbols: list[str] = []
    for symbol in chosen_symbols:
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)

    saved_paths: list[Path] = []
    for index, symbol in enumerate(unique_symbols[:3], start=1):
        symbol_trades = trades_df.loc[trades_df["symbol"] == symbol].copy()
        if symbol_trades.empty:
            continue
        trade = symbol_trades.iloc[symbol_trades["net_return"].abs().argmax()]
        output_path = output_dir / f"representative_trade_{index}_{symbol}.png"
        plot_trade_window(signal_bars=signal_bars_by_symbol[symbol], trade=trade, output_path=output_path)
        saved_paths.append(output_path)
    return saved_paths


def plot_trade_window(signal_bars: pd.DataFrame, trade: pd.Series, output_path: Path) -> None:
    entry_time = pd.to_datetime(trade["entry_time"])
    exit_time = pd.to_datetime(trade["exit_time"])
    start_time = entry_time.floor("30min") - pd.Timedelta(hours=20)
    end_time = exit_time.ceil("30min") + pd.Timedelta(hours=10)
    window = signal_bars.loc[(signal_bars.index >= start_time) & (signal_bars.index <= end_time)].copy()
    if window.empty:
        return

    dates = mdates.date2num(window.index.to_pydatetime())
    candle_width = 0.015

    fig, ax = plt.subplots(figsize=(14, 7))
    for x_value, (_, row) in zip(dates, window.iterrows(), strict=False):
        color = "#d7263d" if row["hfq_close"] >= row["hfq_open"] else "#1b998b"
        ax.plot([x_value, x_value], [row["hfq_low"], row["hfq_high"]], color=color, linewidth=1.0)
        body_low = min(row["hfq_open"], row["hfq_close"])
        body_height = max(abs(row["hfq_close"] - row["hfq_open"]), 1e-6)
        rect = Rectangle(
            (x_value - candle_width / 2.0, body_low),
            candle_width,
            body_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.8,
        )
        ax.add_patch(rect)

    entry_pos = min(window.index.searchsorted(entry_time, side="left"), len(window.index) - 1)
    exit_pos = min(window.index.searchsorted(exit_time, side="left"), len(window.index) - 1)
    entry_x = mdates.date2num(window.index[entry_pos].to_pydatetime())
    exit_x = mdates.date2num(window.index[exit_pos].to_pydatetime())

    ax.scatter(entry_x, float(trade["entry_price"]), color="#111827", s=80, marker="^", label="Entry")
    ax.scatter(exit_x, float(trade["exit_price"]), color="#f59e0b", s=80, marker="v", label="Exit")
    ax.set_title(f"{trade['symbol']} {trade['direction']} trade | net return {trade['net_return']:.2%}")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    report_path: Path,
    config: StrategyConfig,
    metrics_df: pd.DataFrame,
    portfolio_metrics: dict[str, float | int],
    period_metrics: list[dict[str, float | int | str]],
    representative_paths: list[Path],
) -> None:
    top_symbols = metrics_df.head(5)
    bottom_symbols = metrics_df.tail(5).sort_values("sharpe")
    best_names = ", ".join(top_symbols["symbol"].tolist())
    weak_names = ", ".join(bottom_symbols["symbol"].tolist())

    report_lines = [
        "# CTA Problem 1 Backtest Report",
        "",
        "## 1. Strategy logic",
        "",
        "- Strategy type: medium-frequency trend-following breakout on each futures symbol independently.",
        f"- Signal bar: `{config.signal_freq}` aggregated from 1-minute data using only `hfq_*` prices.",
        f"- Long entry: `{config.signal_freq}` close breaks above the previous `{config.entry_lookback}` bars' highest close, while fast EMA stays above slow EMA and efficiency ratio >= `{config.er_threshold:.2f}`.",
        "- Short entry: symmetric downside breakout with the same trend-strength filter.",
        f"- Exit: previous `{config.exit_lookback}` bars' breakout failure or fast/slow EMA trend reversal.",
        "- Execution: signals are generated after the signal bar closes; fills use the next available 1-minute `hfq_twap`.",
        "- Cost assumption: open and close both charge `万分之2 = 0.0002` per side.",
        "- Parameter policy: fixed heuristic parameters designed to keep turnover moderate; no per-symbol parameter optimization.",
        "",
        "## 2. Portfolio metrics",
        "",
        "```text",
        format_metrics_block(portfolio_metrics),
        "```",
        "",
        "Year-by-year portfolio metrics:",
        "",
        "```text",
        format_table(pd.DataFrame(period_metrics)[["period", "sharpe", "annualized_return", "max_drawdown", "number_of_transactions"]]),
        "```",
        "",
        "## 3. Symbol cross-section",
        "",
        "Top 5 symbols by Sharpe:",
        "",
        "```text",
        format_table(top_symbols[["symbol", "sharpe", "annualized_return", "max_drawdown", "number_of_transactions"]]),
        "```",
        "",
        "Bottom 5 symbols by Sharpe:",
        "",
        "```text",
        format_table(bottom_symbols[["symbol", "sharpe", "annualized_return", "max_drawdown", "number_of_transactions"]]),
        "```",
        "",
        "## 4. Simple analysis",
        "",
        f"- Cross-section strongest symbols in this run: {best_names}.",
        f"- Cross-section weakest symbols in this run: {weak_names}.",
        "- The strategy works best when a symbol develops persistent directional expansion after consolidation, because the breakout plus EMA alignment filter keeps it in sustained moves and cuts many noisy reversals.",
        "- The strategy struggles in tight oscillating ranges and fast V-shaped reversals, where repeated channel failures still trigger a few stop-and-reverse trades.",
        "- The equal-weight portfolio benefits from diversification because different symbols do not trend at the same time, so the aggregate curve is smoother than many standalone symbols.",
        "",
        "## 5. Output files",
        "",
        "- `charts/symbol_equity_curves.png`: all selected symbols' daily equity curves.",
        "- `charts/portfolio_equity.png`: equal-weight portfolio equity curve.",
        "- `symbol_metrics.csv`: per-symbol performance table.",
        "- `portfolio_metrics.csv`: portfolio summary metrics.",
        "- `all_trades.csv`: all completed trades.",
        "",
        "## 6. Representative trade charts",
        "",
    ]

    if representative_paths:
        for path in representative_paths:
            report_lines.append(f"![{path.stem}](charts/{path.name})")
            report_lines.append("")
    else:
        report_lines.append("No completed trades, so no representative trade charts were generated.")
        report_lines.append("")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def format_metrics_block(metrics: dict[str, float | int]) -> str:
    lines = [
        f"Sharpe ratio: {metrics['sharpe']:.3f}",
        f"Annualized return: {metrics['annualized_return']:.2%}",
        f"Max drawdown: {metrics['max_drawdown']:.2%}",
        f"Calmar ratio: {metrics['calmar']:.3f}",
        f"Win rate: {metrics['win_rate']:.2%}",
        f"Win/loss ratio: {metrics['win_loss_ratio']:.3f}",
        f"Number of transactions: {metrics['number_of_transactions']}",
        f"Average holding time (hours): {metrics['average_holding_hours']:.2f}",
        f"Average profit/loss per transaction: {metrics['average_profit_loss_per_transaction']:.4%}",
    ]
    return "\n".join(lines)


def format_table(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            if "drawdown" in column or "return" in column or "rate" in column:
                display_df[column] = display_df[column].map(lambda x: f"{x:.2%}")
            else:
                display_df[column] = display_df[column].map(lambda x: f"{x:.3f}")
    return display_df.to_string(index=False)
