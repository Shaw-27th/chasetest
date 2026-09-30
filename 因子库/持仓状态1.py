"""
邢不行™️选股框架
Python股票量化投资课程

从外部 xlsx 文件（0MVA_with_持仓状态1.xlsx）读取"持仓状态1"列，
直接作为择时因子使用。

Author: 邢不行
"""

import pandas as pd

# 财务因子列
fin_cols = []

# 外部 xlsx 文件路径（含"持仓状态1"列）
SOURCE_XLSX = r"D:\stock_data\stock-0MVA-data-2026-08-09\0MVA_with_持仓状态1.xlsx"


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    从外部 xlsx 文件读取"持仓状态1"列，并按 stock_df 的日期对齐写入因子列。

    :param df: 单只股票的日线数据
    :param param: 预留参数
    :param kwargs: col_name=输出因子列名
    :return: 只包含新因子列的 DataFrame
    """
    col_name = kwargs["col_name"]

    src = pd.read_excel(SOURCE_XLSX)
    src["日期"] = pd.to_datetime(src["日期"])
    if "持仓状态1" not in src.columns:
        raise KeyError(
            f"外部 xlsx ({SOURCE_XLSX}) 缺少 '持仓状态1' 列。"
            f"请确认文件已正确生成。实际列名: {src.columns.tolist()}"
        )

    date_col = "交易日期" if "交易日期" in df.columns else "日期"
    src = src.set_index("日期").reindex(pd.to_datetime(df[date_col]))

    df[col_name] = src["持仓状态1"].fillna(0).astype("float64").values

    return df[[col_name]]