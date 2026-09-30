"""读取外部活筹收盘值，计算指定周期的简单移动平均；默认周期为 10。

配置示例：("活筹MA", True, 10, 1)。
仅提供指标，买卖规则由信号库实现。
"""

from pathlib import Path

import pandas as pd


fin_cols = []
SOURCE_XLSX = Path(__file__).resolve().parents[1] / "check" / "0MVA_with_持仓状态1.xlsx"


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """先在完整活筹历史上计算均线，再按交易日期对齐，不改变输入行序或索引。

    前周期不足的日期及来源缺失的日期保留 NaN，不填充为零。
    可通过 source_xlsx 关键字参数指定其他来源文件。
    """
    period = 10 if param is None else param
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise ValueError("活筹MA周期必须为正整数，例如 10")

    col_name = kwargs["col_name"]
    date_col = "交易日期" if "交易日期" in df.columns else "日期"
    if date_col not in df.columns:
        raise KeyError("输入行情缺少交易日期或日期列")

    source_path = Path(kwargs.get("source_xlsx", SOURCE_XLSX))
    source = pd.read_excel(source_path, sheet_name="0AMV日线", usecols=["日期", "收盘"])
    dates = source["日期"]
    if pd.api.types.is_numeric_dtype(dates):
        # 兼容未设置日期格式、以 Excel 日期序号保存的来源。
        source["日期"] = pd.to_datetime(dates, unit="D", origin="1899-12-30")
    else:
        source["日期"] = pd.to_datetime(dates, errors="raise")
    source["日期"] = source["日期"].dt.normalize()
    if source["日期"].isna().any() or source["日期"].duplicated().any():
        raise ValueError("活筹数据日期存在空值或重复，请先检查来源文件")

    source["收盘"] = pd.to_numeric(source["收盘"], errors="raise")
    if source["收盘"].isna().any():
        raise ValueError("活筹数据收盘值存在空值，请先检查来源文件")
    source = source.sort_values("日期").set_index("日期")
    moving_average = source["收盘"].rolling(window=period, min_periods=period).mean()

    target_dates = pd.to_datetime(df[date_col], errors="raise").dt.normalize()
    values = moving_average.reindex(pd.DatetimeIndex(target_dates)).to_numpy()
    return pd.DataFrame({col_name: values}, index=df.index)
