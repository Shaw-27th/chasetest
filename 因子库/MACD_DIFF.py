"""
MACD 快慢线离差 DIFF = EMA(fast) - EMA(slow)

param: (快线周期, 慢线周期, 信号线周期)，例如 (12, 26, 9)
signal_n 不参与 DIFF 计算，保留在参数里是为了与 MACD_DEA 共用同一组参数。
"""

import pandas as pd

fin_cols = []  # 财务因子列󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿


# noinspection PyUnusedLocal󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
def add_factor(df: pd.DataFrame, param, fin_data=None, **kwargs) -> pd.DataFrame:
    """
    计算 MACD 快慢线离差 DIFF，只返回包含因子列的 DataFrame。

    框架不再做周期聚合（transfer_to_period_data 已删除），因子只按日线逐行输出，
    换仓日期由选股/择时阶段统一处理；本因子服务于个股择时的 MACD 穿越判断。
    """
    fast_n, slow_n, _ = (int(p) for p in param)
    col_name = kwargs["col_name"]

    close = df["收盘价_复权"]
    dif = close.ewm(span=fast_n, adjust=False).mean() - close.ewm(span=slow_n, adjust=False).mean()

    factor_df = pd.DataFrame({col_name: dif}, index=df.index)
    return factor_df
