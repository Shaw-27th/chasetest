"""
个股日线 活跃市值1（活跃市值择时）。

本文件只负责"从指标列判方向"，指标本身由 因子库/收盘价.py 计算。

** 配置要求 **
择时策略的 timing.factor_list 必须包含一个收盘价因子；
信号函数从收盘价序列计算涨跌幅，再判金叉买、死叉卖。
本信号不做提前校验，配错（缺因子）会以自然的 KeyError 形式暴露。

** 策略逻辑 **
- 买入条件：当日+前一日涨跌幅合计 >= buy_2day_thresh（默认 4.0%）
            单日涨幅无论多大都不算，必须两日内合计涨幅达到阈值才买入
- 卖出条件：当日涨跌幅 < sell_thresh（默认 -2.3%） -> 立刻空仓
- 卖出后维持空仓，直到下一个买入信号触发才再次持有

** params 说明（全部可省略，取默认值）**
  - buy_2day_thresh: 默认 4.0。  两日合计涨幅阈值（单位 %）
  - sell_thresh:     默认 -2.3。 卖出跌幅阈值（单位 %）
  - mode:            "cross"（默认）只在条件首次成立那天产生买卖事件，其余日期维持原持仓；
                              "state" 逐日判断：满足买入条件则持有，否则空仓
  - confirm_n:       默认 1。   条件需连续成立 n 个交易日才确认
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
    :param stock_df: 单标的日线数据，已含 factor_list 对应的因子列
    :return: 与 stock_df 等长的 0/1 持仓状态 Series，1 表示持有，0 表示空仓
    """
    params = timing_config.timing.params
    buy_2day_thresh = float(_get_param(params, "buy_2day_thresh", 4.0))
    sell_thresh = float(_get_param(params, "sell_thresh", -2.3))
    mode = _get_param(params, "mode", "cross")
    confirm_n = int(_get_param(params, "confirm_n", 1))

    # 按因子名从 factor_list 中定位实际列名；缺因子/缺列会自然抛 KeyError
    factor_list = timing_config.timing.factor_list
    if len(factor_list) != 1:
        raise ValueError("活跃市值1要求timing.factor_list恰好声明一个收盘价因子")
    close = stock_df[factor_list[0].col_name]

    # 涨跌幅（百分比）
    change_pct = close.pct_change() * 100

    # 多空原始条件。
    # 买入：两日合计涨跌幅 >= buy_2day_thresh（单日大涨不算，必须两日内累计达到阈值）
    two_day_sum = change_pct.fillna(0) + change_pct.shift(1).fillna(0)
    long_con = two_day_sum >= buy_2day_thresh
    # 卖出：单日涨跌幅 < sell_thresh
    short_con = change_pct < sell_thresh

    # 预热期（首日无涨跌幅）不产生任何条件
    valid = change_pct.notna()
    long_con &= valid
    short_con &= valid

    if confirm_n > 1:
        long_con = long_con.rolling(confirm_n).sum().eq(confirm_n)
        short_con = short_con.rolling(confirm_n).sum().eq(confirm_n)

    if mode == "state":
        # 逐日判断：只有站在买入条件上才持有，否则空仓
        position = long_con.astype("float64").rename("持仓状态")
    else:
        # cross：只在条件首次成立那天产生买卖事件，其余日期维持原持仓
        signal = pd.Series(np.nan, index=stock_df.index)
        signal[long_con & ~long_con.shift(1, fill_value=False)] = 1.0
        signal[short_con & ~short_con.shift(1, fill_value=False)] = 0.0
        position = signal.ffill().fillna(0.0).rename("持仓状态")

    # ========== 实时打印每天的日期和持仓状态 ==========
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
    print("【活跃市值1】每日持仓状态  1=持仓, 0=空仓")
    print(f"  总天数={len(stock_df)}  持仓天数={int((position == 1).sum())}  "
          f"空仓天数={int((position == 0).sum())}  "
          f"买入={int(is_buy.sum())}次  卖出={int(is_sell.sum())}次")
    print("=" * 80)
    print(f"  {'日期':<12} {'涨跌幅%':>9} {'2日合计%':>10} {'信号':<7} 状态")
    print("-" * 80)

    for i in range(len(stock_df)):
        d = date_series.iloc[i]
        try:
            d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        except Exception:
            d_str = str(d)[:10]
        pct = change_pct.iloc[i]
        tds = two_day_sum.iloc[i]
        pos = int(position.iloc[i])
        if is_buy.iloc[i]:
            sig = "[BUY]"
        elif is_sell.iloc[i]:
            sig = "[SELL]"
        else:
            sig = "  -"
        pos_str = "1持仓" if pos == 1 else "0空仓"
        pct_str = f"{pct:>+8.2f}%" if pd.notna(pct) else "     nan"
        tds_str = f"{tds:>+9.2f}%" if pd.notna(tds) else "      nan"
        print(f"  {d_str:<12} {pct_str} {tds_str} {sig:<7} {pos_str}")

    print("=" * 80)
    last_pos = int(position.iloc[-1])
    last_d = date_series.iloc[-1]
    print(f"  【最新】{pd.Timestamp(last_d).strftime('%Y-%m-%d')} 持仓状态={last_pos} "
          f"({'持仓中' if last_pos == 1 else '空仓中'})")
    print("=" * 80)

    return position