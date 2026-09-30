"""
MA10均线择时。

本文件从 xlsx 文件读取日线数据，直接用收盘价计算 MA10 进行择时：
  - 收盘价 > MA10 → 持有（1）
  - 收盘价 < MA10 → 空仓（0）

** params 说明（全部可省略，取默认值）**
  - mode: "cross"（默认）金叉买、死叉卖，死区内维持原持仓；"state" 逐日判断，不满足做多条件即空仓
  - confirm_n: 默认 1。条件需连续成立 n 个交易日才确认，用延迟换胜率
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.model.timing_signal import TimingConfig


def _get_param(params, key, default):
    """
    从 params 读取参数，缺失时返回默认值。
    """
    if isinstance(params, dict):
        return params.get(key, default)
    try:
        return params[key]
    except (KeyError, TypeError):
        return default


def timing_signal(timing_config: "TimingConfig", stock_df: pd.DataFrame) -> pd.Series:
    """
    :param timing_config: TimingConfig，信号从 timing_config.timing.factor_list 按因子名定位列名、
                          从 timing_config.timing.params 读取参数
    :param stock_df: 单标的日线数据，必须包含 收盘价 列
    :return: 与 stock_df 等长的 0/1 持仓状态 Series，1 表示持有，0 表示空仓
    """
    params = timing_config.timing.params
    mode = _get_param(params, "mode", "cross")
    confirm_n = int(_get_param(params, "confirm_n", 1))

    # 读取收盘价列（兼容 "收盘价" 和 "收盘" 两种列名）
    close_col = None
    for col_name in ["收盘价", "收盘", "close"]:
        if col_name in stock_df.columns:
            close_col = col_name
            break
    if close_col is None:
        raise KeyError(f"数据中找不到收盘价列，可用列：{stock_df.columns.tolist()}")

    close = stock_df[close_col]

    # 计算 MA10
    ma10 = close.rolling(window=10, min_periods=1).mean()

    # 多空原始条件
    long_con = close > ma10   # 收盘价在 MA10 上方
    short_con = close < ma10  # 收盘价在 MA10 下方

    # 指标预热期不产生任何条件
    valid = close.notna()
    long_con &= valid
    short_con &= valid

    if confirm_n > 1:
        long_con = long_con.rolling(confirm_n).sum().eq(confirm_n)
        short_con = short_con.rolling(confirm_n).sum().eq(confirm_n)

    if mode == "state":
        # 逐日判断：收盘价在 MA10 上方则持有，否则空仓
        return long_con.astype("float64").rename("持仓状态")

    # cross：只在条件首次成立那天产生买卖事件，其余日期维持原持仓
    signal = pd.Series(np.nan, index=stock_df.index)
    signal[long_con & ~long_con.shift(1, fill_value=False)] = 1.0   # 金叉（买入）
    signal[short_con & ~short_con.shift(1, fill_value=False)] = 0.0  # 死叉（卖出）
    return signal.ffill().fillna(0.0).rename("持仓状态")
