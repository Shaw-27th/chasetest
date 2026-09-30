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
fin_cols = [
    "R_np_atoopc@xbx_单季",
    "B_total_equity_atoopc@xbx",
    "R_np_atoopc@xbx_ttm",
]  # 财务因子列，配置后系统会自动加载对应的财务数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿


def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
    """
    计算 ROE 因子，并返回包含因子列的 DataFrame。

    工作流程：
    1. 根据 ``param`` 选择全年或单季净利润字段。
    2. 使用净利润除以归属于母公司所有者权益，得到 ROE。

    :param df: 单只股票的日线数据，需包含对应净利润和归母所有者权益字段。
    :param param: 利润口径，支持 ``全年`` 和 ``单季``。
    :param kwargs: 其他参数，其中 ``col_name`` 为输出因子列名。
    :return: 只包含新因子列的 DataFrame，行数和索引与输入 df 一致。

    注意：财务字段由 ``fin_cols`` 声明，框架会在计算因子前自动合并到日线数据中。
    """

    # ======================== 参数处理 ===========================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 从kwargs中提取因子列的名称，这里使用'col_name'来标识因子列名称󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    col_name = kwargs["col_name"]

    # 净利润相关字段说明󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # - R_np_atoopc@xbx_ttm:利润表的归属于母公司所有者的净利润ttm󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # - R_np_atoopc@xbx_单季:利润表的归属于母公司所有者的净利润单季度󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    profit_cols = {"全年": "R_np_atoopc@xbx_ttm", "单季": "R_np_atoopc@xbx_单季"}

    # 根据param选择相应的净利润字段󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if param not in profit_cols:
        raise ValueError(f"ROE因子不支持的参数值：{param}")
    else:
        profit_col = profit_cols[param]

    # ======================== 计算因子 ===========================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # ROE：净资产收益率 = 净利润 / 净资产󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # - B_total_equity_atoopc@xbx:资产负债表_所有者权益的归属于母公司所有者权益合计󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df[col_name] = df[profit_col] / df["B_total_equity_atoopc@xbx"]

    return df[[col_name]]
