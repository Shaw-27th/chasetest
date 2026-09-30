"""从活筹 Excel 读取收盘值，独立于交易标的的行情收盘价。"""

import pandas as pd

from 因子库.活筹MA import SOURCE_XLSX, align_factor, read_active_close

fin_cols = []


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """返回按交易日期对齐的活筹收盘因子，缺失日期保留 NaN。"""
    close = read_active_close(kwargs.get("source_xlsx", SOURCE_XLSX))
    return align_factor(df, close, kwargs["col_name"])
