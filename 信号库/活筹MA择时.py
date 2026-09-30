"""活筹收盘高于均线持仓，低于均线空仓，相等时保持之前状态。

依赖活筹收盘、活筹MA两个因子；均线预热期为空仓。
只输出当日目标持仓，不在这里移动信号日期，成交时点由框架决定。
"""

import pandas as pd


def timing_signal(timing_config, stock_df: pd.DataFrame) -> pd.Series:
    """逐日比较指标，返回与输入等长、同索引的 0/1 持仓状态。"""
    factors = timing_config.timing.factor_list
    if len(factors) != 2 or {factor.name for factor in factors} != {"活筹收盘", "活筹MA"}:
        raise ValueError("活筹MA择时必须恰好配置活筹收盘和活筹MA两个因子")
    columns = {factor.name: factor.col_name for factor in factors}
    close = pd.to_numeric(stock_df[columns["活筹收盘"]], errors="raise")
    ma = pd.to_numeric(stock_df[columns["活筹MA"]], errors="raise")
    dates = pd.to_datetime(stock_df["交易日期"], errors="raise")
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("择时交易日期必须有效、递增且无重复")
    if close.isna().any():
        raise ValueError("交易日期缺少活筹收盘数据，请检查 Excel 日期覆盖范围")
    valid = ma.notna()
    if valid.any() and ma.loc[valid.cummax()].isna().any():
        raise ValueError("均线预热完成后出现缺失值，请检查活筹MA数据")
    state = pd.Series(float("nan"), index=stock_df.index, dtype="float64")
    state.loc[~valid | (close < ma)] = 0.0
    state.loc[valid & (close > ma)] = 1.0
    return state.ffill().fillna(0.0).rename("持仓状态")
