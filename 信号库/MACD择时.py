"""
个股日线 MACD 择时。

本文件只负责"从指标列判方向"，指标本身由 因子库/MACD_DIFF.py 和 因子库/MACD_DEA.py 计算。

** 配置要求 **
择时策略的 timing.factor_list 必须包含 MACD_DIFF 和 MACD_DEA 两个因子，且参数完全一致
（否则 DIFF/DEA 不同源，金叉死叉判断没有意义）；信号函数按因子名自行定位实际列名。
本信号不做提前校验，配错（缺因子、参数不一致）会以自然的 KeyError/错误信号形式暴露。

** params 说明（全部可省略，取默认值）**
  - mode: "cross"（默认）金叉买、死叉卖，死区内维持原持仓；"state" 逐日判断，不满足做多条件即空仓
  - zero_filter: 默认 False。True 表示只在零轴上方（DIFF>0）认金叉、零轴下方（DIFF<0）认死叉
  - min_bar: 默认 0.0。柱值 DIFF-DEA 需越过 ±min_bar 才算穿越，用于过滤震荡市的贴线反复
  - confirm_n: 默认 1。条件需连续成立 n 个交易日才确认，用延迟换胜率
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.model.timing_signal import TimingConfig


def timing_signal(timing_config: "TimingConfig", stock_df: pd.DataFrame) -> pd.Series:
    """
    :param timing_config: TimingConfig，信号从 timing_config.timing.factor_list 按因子名定位列名、
                          从 timing_config.timing.params 读取参数
    :param stock_df: 单标的日线数据，已含 factor_list 对应的因子列
    :return: 与 stock_df 等长的 0/1 持仓状态 Series，1 表示持有，0 表示空仓
    """
    params = timing_config.timing.params
    mode = params["mode"]
    zero_filter = params["zero_filter"]
    min_bar = params["min_bar"]
    confirm_n = params["confirm_n"]

    # 按因子名从 factor_list 中定位实际列名；缺因子/缺列会自然抛 KeyError󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    cols = {factor.name: factor.col_name for factor in timing_config.timing.factor_list}
    dif = stock_df[cols["MACD_DIFF"]]
    dea = stock_df[cols["MACD_DEA"]]
    bar = dif - dea

    # 多空原始条件。min_bar 在两者之间留出一段死区，两个条件互斥且可以同时为假。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    long_con = bar > min_bar
    short_con = bar < -min_bar
    if zero_filter:
        long_con &= dif > 0
        short_con &= dif < 0

    # 指标预热期（EMA 尚未有值）不产生任何条件󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    valid = dif.notna() & dea.notna()
    long_con &= valid
    short_con &= valid

    if confirm_n > 1:
        long_con = long_con.rolling(confirm_n).sum().eq(confirm_n)
        short_con = short_con.rolling(confirm_n).sum().eq(confirm_n)

    if mode == "state":
        # 逐日判断：只有站在做多条件上才持有，死区和预热期都算空仓󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        return long_con.astype("float64").rename("持仓状态")

    # cross：只在条件首次成立那天产生买卖事件，其余日期维持原持仓󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    signal = pd.Series(np.nan, index=stock_df.index)
    signal[long_con & ~long_con.shift(1, fill_value=False)] = 1.0
    signal[short_con & ~short_con.shift(1, fill_value=False)] = 0.0
    return signal.ffill().fillna(0.0).rename("持仓状态")
