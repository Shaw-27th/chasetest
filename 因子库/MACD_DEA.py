"""
MACD 信号线 DEA = EMA(DIFF, signal)

param: (快线周期, 慢线周期, 信号线周期)，例如 (12, 26, 9)
必须与 MACD_DIFF 使用完全相同的参数，否则 DIFF/DEA 不同源，穿越判断无意义。
"""

import pandas as pd

fin_cols = []  # 财务因子列󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿


# noinspection PyUnusedLocal󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
def add_factor(df: pd.DataFrame, param, fin_data=None, **kwargs) -> pd.DataFrame:
    """
    计算 MACD 信号线 DEA，只返回包含因子列的 DataFrame。

    框架不再做周期聚合（transfer_to_period_data 已删除），因子只按日线逐行输出，
    换仓日期由选股/择时阶段统一处理；本因子服务于个股择时的 MACD 穿越判断。
    """
    fast_n, slow_n, signal_n = (int(p) for p in param)
    col_name = kwargs["col_name"]

    close = df["收盘价_复权"]
    dif = close.ewm(span=fast_n, adjust=False).mean() - close.ewm(span=slow_n, adjust=False).mean()
    dea = dif.ewm(span=signal_n, adjust=False).mean()

    factor_df = pd.DataFrame({col_name: dea}, index=df.index)
    return factor_df
