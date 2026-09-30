"""
活跃市值4（直接读取外部 xlsx 的"持仓状态1"列）。

本文件直接读取 stock_df 中由"持仓状态1"因子提供的列（值：0=空仓, 1=持仓），
不做任何计算，直接返回作为择时信号。

** 配置要求 **
择时策略的 timing.factor_list 必须恰好声明一个名为"持仓状态1"的因子；
该因子由 因子库/持仓状态1.py 提供，从外部 xlsx 文件读取。
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.model.timing_signal import TimingConfig



def timing_signal(timing_config: "TimingConfig", stock_df: pd.DataFrame) -> pd.Series:
    """
    :param timing_config: TimingConfig
    :param stock_df: 单标的日线数据，已含"持仓状态1"因子列
    :return: 与 stock_df 等长的 0/1 持仓状态 Series
    """
    factor_list = timing_config.timing.factor_list
    if len(factor_list) != 1:
        raise ValueError("活跃市值4要求timing.factor_list恰好声明一个'持仓状态1'因子")

    target_col = factor_list[0].col_name
    if target_col not in stock_df.columns:
        raise KeyError(
            f"stock_df 缺少因子列 {target_col}。"
            f"请确认 factor_list 中因子名 '持仓状态1' 与 因子库/持仓状态1.py 一致，"
            f"并已重新运行 step2。"
            f"可用列: {stock_df.columns.tolist()}"
        )

    position = stock_df[target_col].astype("float64").fillna(0.0)
    position = position.where(position.isin([0.0, 1.0]), 0.0)
    position = position.rename("持仓状态")

    date_col = None
    for col in ("交易日期", "日期", "datetime", "date", "candle_end_time"):
        if col in stock_df.columns:
            date_col = col
            break
    if date_col is None:
        date_series = pd.Series(stock_df.index, index=stock_df.index)
    else:
        date_series = stock_df[date_col]

    is_buy = (position == 1) & (position.shift(1, fill_value=0) == 0)
    is_sell = (position == 0) & (position.shift(1, fill_value=0) == 1)

    print("\n" + "=" * 80)
    print("【活跃市值4】每日持仓状态（直接读取 xlsx 的'持仓状态1'列） 1=持仓, 0=空仓")
    print(f"  总天数={len(stock_df)}  持仓天数={int((position == 1).sum())}  "
          f"空仓天数={int((position == 0).sum())}  "
          f"买入={int(is_buy.sum())}次  卖出={int(is_sell.sum())}次")
    print("=" * 80)
    print(f"  {'日期':<12} {'持仓状态(1/0)':<14} {'变化':<8} {'持仓':<6}")
    print("-" * 80)

    for i in range(len(stock_df)):
        d = date_series.iloc[i]
        try:
            d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        except Exception:
            d_str = str(d)[:10]
        pos = int(position.iloc[i])
        if is_buy.iloc[i]:
            change = "[BUY]"
        elif is_sell.iloc[i]:
            change = "[SELL]"
        else:
            change = "—"
        pos_str = "1 持仓" if pos == 1 else "0 空仓"
        print(f"  {d_str:<12} {pos:<14} {change:<8} {pos_str}")

    print("=" * 80)
    last_pos = int(position.iloc[-1])
    last_d = date_series.iloc[-1]
    print(f"  【最新】{pd.Timestamp(last_d).strftime('%Y-%m-%d')} 持仓状态={last_pos} "
          f"({'持仓中' if last_pos == 1 else '空仓中'})")
    print("=" * 80)

    return position