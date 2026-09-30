"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import pandas as pd

# 财务因子列：此列表用于存储财务因子相关的列名称󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
fin_cols = []  # 财务因子列，配置后系统会自动加载对应的财务数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    使用当日收盘价计算因子，并返回包含因子列的 DataFrame。

    :param df: 单只股票的日线数据，需包含收盘价列（兼容 "收盘价" / "收盘" / "close"）。
    :param param: 预留的因子参数，当前计算不使用。
    :param kwargs: 其他参数，其中 ``col_name`` 为输出因子列名。
    :return: 只包含新因子列的 DataFrame，行数和索引与输入 df 一致。
    """

    # ======================== 参数处理 ===========================
    # 从kwargs中提取因子列的名称，这里使用'col_name'来标识因子列名称
    col_name = kwargs["col_name"]

    # ======================== 识别收盘价列 ===========================
    # 优先使用 "收盘价"，其次 "收盘"、"close"、"Close"，找不到再抛错
    close_col = None
    for cand in ("收盘价", "收盘", "close", "Close"):
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        raise KeyError(
            f"收盘价因子找不到收盘价列，已尝试 收盘价 / 收盘 / close / Close，"
            f"可用列：{df.columns.tolist()}"
        )

    # ======================== 计算因子 ===========================
    # 使用当日收盘价作为因子值
    df[col_name] = df[close_col]

    return df[[col_name]]