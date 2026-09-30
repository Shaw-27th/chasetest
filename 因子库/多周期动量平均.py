"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import pandas as pd

fin_cols = []  # 财务因子列󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿


# noinspection PyUnusedLocal󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
def add_factor(df: pd.DataFrame, param, fin_data=None, **kwargs) -> pd.DataFrame:
    """
    合并数据后计算策略所需的因子。

    :param df: 输入的K线数据，包含各类市场指标。
    :param param: 策略参数，用于配置因子计算的具体细节。
    :param fin_data: 财务数据字典，格式为 {'财务数据': fin_df, '原始财务数据': raw_fin_df}，
                     其中raw_fin_df包含需要舍弃的原始报告数据。
    :param kwargs: 其他关键字参数，包括但不限于因子名称（'col_name'）。
    :return: 包含计算后因子的 DataFrame，索引与输入 df 一致。
    """

    # ====================================== 参数处理 ==========================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 从额外参数中获取因子名称󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    col_name = kwargs["col_name"]
    windows = list(param) # 要计算的动量窗口，例如[5，10，20，40]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # ====================================== 计算因子 ==========================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 每个窗口各算一次涨跌幅，横向取平均，得到综合动量󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    momentum = pd.concat(
        [df["收盘价_复权"].pct_change(w) for w in windows],
        axis=1,
    )
    # skipna = False : 所有窗口都凑齐历史后才出值，避免预热期出现“部分窗口平均”的假值󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df[col_name] = momentum.mean(axis=1, skipna=False)

    return df[[col_name]]
