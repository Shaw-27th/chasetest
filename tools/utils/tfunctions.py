"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

"""
—————‌——‌——‌—‌—‌—‌—‌———‌—‌—‌—‌—‌——‌—‌———————‌———‌—‌———‌—‌—‌—‌
数据处理函数集 — 表格数据加载、聚合与多进程处理工具。
—————‌——‌——‌—‌—‌—‌—‌———‌—‌—‌—‌—‌——‌—‌———————‌———‌—‌———‌—‌—‌—‌
"""

import hashlib
import math
import os
from pathlib import Path
import pandas as pd

from core.model.strategy_config import filter_series_by_range


# region 通用函数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
def list_factor_cache_names(cache_dir: Path) -> list[str]:
    """列出运行缓存中的日线因子文件名 stem，兼容 factor_*.parquet 和 factor_*.pkl。"""
    factor_names = set()
    for file in Path(cache_dir).iterdir():
        if not file.is_file() or file.suffix not in [".parquet", ".pkl"]:
            continue
        if not file.name.startswith("factor_") or file.name.startswith(("factor_hour_", "factor_hour_v2_")):
            continue
        factor_names.add(file.stem)
    return sorted(factor_names)


def get_data_path_md5(data_path):
    # 将文件夹的大小、更改时间信息作为参数，生成md5值󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    info_txt = ""
    if data_path.is_dir():
        files = list(sorted([str(f) for f in data_path.iterdir() if f.is_file()]))
        for file in files:
            info_txt += f"{os.path.getsize(file)}-{os.path.getmtime(file)}"
    if data_path.is_file():
        info_txt += f"{os.path.getsize(data_path)}-{os.path.getmtime(data_path)}"
    md5_txt = hashlib.md5(info_txt.encode("utf-8")).hexdigest()
    return md5_txt


def read_txt(path):
    with open(path, "r") as f:
        txt = f.read()
    return txt


def write_txt(path, txt):
    with open(path, "w") as f:
        f.write(txt)


def filter_stock(df):
    """
    过滤函数，ST/退市/交易天数不足等情况
    :param df:
    :return:
    """
    # =删除不能交易的周期数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 删除月末为st状态的周期数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df = df[df["股票名称"].str.contains("ST") == False]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # # 删除月末为s状态的周期数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df = df[df["股票名称"].str.contains("S") == False]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # # 删除月末有退市风险的周期数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df = df[df["股票名称"].str.contains("\*") == False]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df = df[df["股票名称"].str.contains("退") == False]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    #󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df = df[df["下日_是否交易"] == 1]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df = df[df["下日_开盘涨停"] == False]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df = df[df["下日_是否ST"] == False]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df = df[df["下日_是否退市"] == False]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df = df[df["上市至今交易天数"] > 250]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    cond1 = ~df["股票名称"].str.contains("ST", regex=False, na=False)
    # 删除月末为s状态的周期数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    cond2 = ~df["股票名称"].str.contains("S", regex=False, na=False)
    # 删除月末有退市风险的周期数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    cond3 = ~df["股票名称"].str.contains("*", regex=False, na=False)
    cond4 = ~df["股票名称"].str.contains("退", regex=False, na=False)

    cond6 = df["下日_是否交易"] == 1
    cond7 = df["下日_开盘涨停"] == False
    cond8 = df["下日_是否ST"] == False
    cond9 = df["下日_是否退市"] == False

    cond10 = df["上市至今交易天数"] > 250
    common_filter = cond1 & cond2 & cond3 & cond4 & cond6 & cond7 & cond8 & cond9 & cond10
    df = df.loc[common_filter]

    return df


def float_num_process(num, return_type=float, keep=2, max=5):
    """
    针对绝对值小于1的数字进行特殊处理，保留非0的N位（N默认为2，即keep参数）
    输入  0.231  输出  0.23
    输入  0.0231  输出  0.023
    输入  0.00231  输出  0.0023
    如果前面max个都是0，直接返回0.0
    :param num: 输入的数据
    :param return_type: 返回的数据类型，默认是float
    :param keep: 需要保留的非零位数
    :param max: 最长保留多少位
    :return:
        返回一个float或str
    """

    # 如果输入的数据是0，直接返回0.0󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if num == 0.0:
        return 0.0

    # 绝对值大于1的数直接保留对应的位数输出󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if abs(num) > 1:
        return round(num, keep)
    # 获取小数点后面有多少个0󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    zero_count = -int(math.log10(abs(num)))
    # 实际需要保留的位数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    keep = min(zero_count + keep, max)

    # 如果指定return_type是float，则返回float类型的数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if return_type == float:
        return round(num, keep)
    # 如果指定return_type是str，则返回str类型的数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    else:
        return str(round(num, keep))


# endregion󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿


def read_period_and_offset_file(file_path):
    """
    载入周期offset文件
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="gbk", parse_dates=["交易日期"], skiprows=1)
        return df
    else:
        print(f"文件{file_path}不存在，请获取period_offset.csv文件后再试")
        raise FileNotFoundError("文件不存在")


def import_index_data(path, date_range=(None, None), max_param=0):
    """
    导入指数数据并进行预处理

    参数:
    path (str): 指数数据文件的路径
    date_range (list, optional): 回测的时间范围，格式为 [开始日期, 结束日期]，默认为 [None, None]
    max_param (int, optional): 因子的最大周期数，用于控制开始日期，确保rolling类因子，前置数据不是NaN，默认为 0

    返回:
    DataFrame: 处理后的指数数据，包含交易日期和指数涨跌幅
    """
    # 导入指数数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df_index = pd.read_csv(path, parse_dates=["candle_end_time"], encoding="gbk")

    # 计算涨跌幅󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df_index["指数涨跌幅"] = df_index["close"].pct_change()
    # 第一天的指数涨跌幅是开盘买入的涨跌幅󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df_index["指数涨跌幅"] = df_index["指数涨跌幅"].fillna(value=df_index["close"] / df_index["open"] - 1)

    # 去除涨跌幅为空的行󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df_index.dropna(subset=["指数涨跌幅"], inplace=True)

    # 重命名列󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df_index.rename(columns={"candle_end_time": "交易日期"}, inplace=True)

    # 根据日期范围过滤数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if date_range[0]:
        if max_param == 0:
            df_index = df_index[df_index["交易日期"] >= pd.to_datetime(date_range[0])]
            # print(f'💡 回测开始时间：{df_index["交易日期"].iloc[0].strftime("%Y-%m-%d")}')󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 当提供了周期数之后󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        else:
            # 计算新的开始日期󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            start_index = df_index[df_index["交易日期"] >= pd.to_datetime(date_range[0])].index[0]
            start_date = df_index["交易日期"][start_index].strftime("%Y-%m-%d")

            # 移动周期，获取可以让因子数值不为Nan的开始日期󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            shifted_date = df_index["交易日期"].shift(max_param)
            shifted_date.bfill(inplace=True)  # 前置数据不是NaN󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

            # 过滤前置数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            df_index = df_index[df_index["交易日期"] >= shifted_date[start_index]]
            new_start_date = df_index["交易日期"].iloc[0].strftime("%Y-%m-%d")
            print(f"💡 回测开始时间：{start_date}，移动{max_param}个周期，最新交易日：{new_start_date}")
    if date_range[1]:
        df_index = df_index[df_index["交易日期"] <= pd.to_datetime(date_range[1])]
        # print(f'回测结束时间：{df_index["交易日期"].iloc[-1].strftime("%Y-%m-%d")}')󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # 按时间排序并重置索引󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df_index.sort_values(by=["交易日期"], inplace=True)
    df_index.reset_index(inplace=True, drop=True)

    return df_index


def filter_common(df, filter_list):
    condition = pd.Series(True, index=df.index)

    for filter_config in filter_list:
        col_name = f"factor_{filter_config.col_name}"
        match filter_config.method.how:
            case "rank":
                rank = df.groupby("交易日期")[col_name].rank(ascending=filter_config.is_sort_asc, pct=False)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case "pct":
                rank = df.groupby("交易日期")[col_name].rank(ascending=filter_config.is_sort_asc, pct=True)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case "val":
                condition = condition & filter_series_by_range(df[col_name], filter_config.method.range)
            case _:
                raise ValueError(f"不支持的过滤方式：{filter_config.method.how}")

    return condition
