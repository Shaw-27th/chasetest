"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import json
import os
import random
from decimal import ROUND_UP
import time

import numpy as np
import pandas as pd
import requests

from core.figure import draw_equity_curve_plotly

pd.set_option("expand_frame_repr", False)
pd.set_option("future.no_silent_downcasting", True)
# print输出中文表头对齐󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)


def cal_fuquan_price(df, fuquan_type="后复权", method=None):
    """
    用于计算复权价格

    参数:
    df (DataFrame): 必须包含的字段：收盘价，前收盘价，开盘价，最高价，最低价
    fuquan_type (str, optional): 复权类型，可选值为 '前复权' 或 '后复权'，默认为 '后复权'
    method (str, optional): 额外计算复权价格的方法，如 '开盘'，默认为 None

    返回:
    DataFrame: 最终输出的df中，新增字段：收盘价_复权，开盘价_复权，最高价_复权，最低价_复权
    """

    # 计算复权因子󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    fq_factor = (df["收盘价"] / df["前收盘价"]).cumprod()

    # 计算前复权或后复权收盘价󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if fuquan_type == "后复权":  # 如果使用后复权方法󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        fq_close = fq_factor * (df.iloc[0]["收盘价"] / fq_factor.iloc[0])
    elif fuquan_type == "前复权":  # 如果使用前复权方法󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        fq_close = fq_factor * (df.iloc[-1]["收盘价"] / fq_factor.iloc[-1])
    else:  # 如果给的复权方法非上述两种标准方法会报错󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        raise ValueError(f"计算复权价时，出现未知的复权类型：{fuquan_type}")

    # 计算其他价格的复权值󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    fq_open = df["开盘价"] / df["收盘价"] * fq_close
    fq_high = df["最高价"] / df["收盘价"] * fq_close
    fq_low = df["最低价"] / df["收盘价"] * fq_close

    # 一次性赋值，提高计算效率󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df = df.assign(
        复权因子=fq_factor, 收盘价_复权=fq_close, 开盘价_复权=fq_open, 最高价_复权=fq_high, 最低价_复权=fq_low
    )

    # 如果指定了额外的方法，计算该方法的复权价格󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if method and method != "开盘":
        df[f"{method}_复权"] = df[method] / df["收盘价"] * fq_close

    # 删除中间变量复权因子󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df.drop(columns=['复权因子'], inplace=True)󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    return df


def get_file_in_folder(path, file_type, contains=None, filters=(), drop_type=False):
    """
    获取指定文件夹下的文件

    参数:
    path (str): 文件夹路径
    file_type (str): 文件类型，例如 '.csv' 或 '.txt'
    contains (str, optional): 文件名中需要包含的字符串，默认为 None
    filters (list, optional): 文件名中需要过滤掉的内容，列表形式，默认为空列表
    drop_type (bool, optional): 是否要去除文件扩展名，默认为 False

    返回:
    list: 符合条件的文件名列表
    """
    # 获取文件夹下的所有文件名󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    file_list = os.listdir(path)

    # 过滤出指定类型的文件󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    file_list = [file for file in file_list if file.endswith(file_type)]

    # 如果指定了包含的字符串，进一步过滤󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if contains:
        file_list = [file for file in file_list if contains in file]

    # 过滤掉指定的内容󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    for con in filters:
        file_list = [file for file in file_list if con not in file]

    # 如果需要去除文件扩展名󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if drop_type:
        file_list = [file[: file.rfind(".")] for file in file_list]

    return file_list


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

    # 保留必要的列󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df_index = df_index[["candle_end_time", "指数涨跌幅"]]

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


def merge_with_index_data(df, index_data, fill_0_list=()):
    """
    原始股票数据在不交易的时候没有数据。
    将原始股票数据和指数数据合并，可以补全原始股票数据没有交易的日期。

    参数:
    df (DataFrame): 股票数据
    index_data (DataFrame): 指数数据
    fill_0_list (list, optional): 合并时需要填充为 0 的字段，默认为空列表

    返回:
    DataFrame: 合并后的股票数据，包含补全的日期
    """
    # 上市日期（合并指标之前记录）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    listing_dt = df["交易日期"].iloc[0]
    max_candle_time = index_data["交易日期"].max()
    # 将股票数据和指数数据合并，结果已经排序󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df = pd.merge(
        left=df,
        right=index_data[index_data["交易日期"] <= max_candle_time],
        on="交易日期",
        how="right",
        sort=True,
        indicator=True,
    )

    # 对开、高、收、低、前收盘价价格进行补全处理󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 用前一天的收盘价，补全收盘价的空值󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    close = df["收盘价"].ffill()
    # 用收盘价补全开盘价、最高价、最低价的空值󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df = df.assign(
        收盘价=close,
        开盘价=df["开盘价"].fillna(value=close),
        最高价=df["最高价"].fillna(value=close),
        最低价=df["最低价"].fillna(value=close),
        均价=df["均价"].fillna(value=close),
        # 补全前收盘价󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        前收盘价=df["前收盘价"].fillna(value=close.shift()),
    )

    # 如果前面算过复权，复权价也做fillna󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if "收盘价_复权" in df.columns:
        fq_cols = dict()
        fq_cols["收盘价_复权"] = df["收盘价_复权"].ffill()
        for col in ["开盘价_复权", "最高价_复权", "最低价_复权"]:
            if col in df.columns:
                fq_cols[col] = df[col].fillna(value=fq_cols["收盘价_复权"])
        df = df.assign(**fq_cols)

    # 将停盘时间的某些列，数据填补为0󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    fill_0_list = list(set(["成交量", "成交额", "涨跌幅"] + fill_0_list))
    df[fill_0_list] = df[fill_0_list].fillna(value=0)

    # 针对特殊的字段需要做特殊空值填充处理󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 填充空值󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    other_cols = [
        "散户资金买入额",
        "中户资金买入额",
        "大户资金买入额",
        "机构资金买入额",
        "散户资金卖出额",
        "中户资金卖出额",
        "大户资金卖出额",
        "机构资金卖出额",
    ]
    existing_cols = [col for col in other_cols if col in df.columns]
    df[existing_cols] = df[existing_cols].fillna(0)

    other_cols = [
        "上证50成分股",
        "沪深300成分股",
        "中证500成分股",
        "中证1000成分股",
        "中证2000成分股",
        "创业板指成分股",
    ]
    existing_cols = [col for col in other_cols if col in df.columns]
    # 创建填充值DataFrame󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    filled_values = df[existing_cols].ffill()
    # 是否停牌󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    is_stop = df["_merge"] == "right_only"
    # 不能直接使用 fillna("N") 和 ffill() 的原因：󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 1. fillna("N")：会误将停牌期间应为"Y"的成分股标记为"N"󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 2. ffill()：会错误延续历史状态，如2022年的"Y"会填充到2023年的NaN，导致已退出成分股的股票仍显示为成分股󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 通过is_stop，控制只有停牌的那几行，才会ffill󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # 使用where进行条件填充：停牌行用ffill值，其他行保持原值。注意，where是cond为False，才会执行filled_values。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df[existing_cols] = df[existing_cols].where(~is_stop, filled_values)
    # 剩余NaN填充为"N"󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df[existing_cols] = df[existing_cols].fillna("N")

    # 未上市的那部分数据是剔除还是保留  25-05-08󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 答：剔除，未上市/退市的数据都是剔除，只有停牌的才保留󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # 去除未上市的数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df = df[df["交易日期"] >= listing_dt]
    # 退市的标识，由数据中心提供󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    delisted_str = "退"
    # 判断是否退市 可以直接用contains，退字在A股里是禁用词，不会出现股票名称含退字的情况󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    is_delisted = df["股票名称"].str.contains(delisted_str, na=False, regex=False)
    temp = df[is_delisted]
    if not temp.empty:
        # 如果发现退市数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        delisted_dt = temp["交易日期"].iloc[-1]
        # 去除退市的数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        df = df[df["交易日期"] <= delisted_dt]
    # 用前一天的数据，补全其余空值（补齐是指停牌的数据，退市/未上市的股票已经在之前剔除了）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df = df.ffill()

    # 剔除nan值数据（sh600002退市日期都不到指数的开始时间，如果不剔除，会导致全为nan的数据出现）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 如果有用户没更新数据（加退字），就会出现上述问题，所以还是先加上，等于是退市的BUG没修󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df = df[df["股票代码"].notnull()]

    # 判断计算当天是否交易󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df["是否交易"] = (df["_merge"] != "right_only").astype(np.int8)  # type: ignore󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    del df["_merge"]
    df.reset_index(drop=True, inplace=True)

    return df


def cal_zdt_price(df, code_type="stock"):
    """
    计算股票当天的涨跌停价格。在计算涨跌停价格的时候，按照严格的四舍五入。
    包含ST股，但是不包含新股。

    涨跌停制度规则:
        ---2020年8月23日
        非ST股票 10%
        ST股票 5%

        ---2020年8月24日至今
        普通非ST股票 10%
        普通ST股票 5%

        ---2026年7月6日至今
        普通非ST股票 10%
        普通ST股票 10%

        科创板（sh68） 20%（一直是20%，不受时间限制）
        创业板（sz3） 20%
        科创板和创业板即使ST，涨跌幅限制也是20%

        北交所（bj） 30%

    参数:
    df (DataFrame): 必须得是日线数据。必须包含的字段：前收盘价，开盘价，最高价，最低价

    返回:
    DataFrame: 包含涨停价、跌停价、一字涨停、一字跌停、开盘涨停、开盘跌停等字段的DataFrame
    """
    from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN

    is_etf = code_type == "etf"

    # ETF 使用 10% 涨跌幅和 0.001 元价格精度；股票继续沿用板块与 ST 规则。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    cond = df["股票名称"].str.contains("ST", regex=False)
    df["涨停价"] = df["前收盘价"] * 1.1
    df["跌停价"] = df["前收盘价"] * 0.9

    # 2026年7月6日之前，ST的涨跌停都是5%，之后，都是10%󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    cond_bj = pd.Series(False, index=df.index)
    if not is_etf:
        new_cond = df["交易日期"] < pd.to_datetime("2026-07-06")
        df.loc[cond & new_cond, "涨停价"] = df["前收盘价"] * 1.05
        df.loc[cond & new_cond, "跌停价"] = df["前收盘价"] * 0.95

        # 计算科创板和新规后的创业板的涨停价和跌停价󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        rule_kcb = df["股票代码"].str.contains("sh68", regex=False)  # 科创板󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        new_rule_cyb = (df["交易日期"] > pd.to_datetime("2020-08-23")) & df["股票代码"].str.contains(
            "sz3", regex=False
        )  # 新规后的创业板󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        merge_rule = rule_kcb | new_rule_cyb
        df.loc[merge_rule, "涨停价"] = df["前收盘价"] * 1.2
        df.loc[merge_rule, "跌停价"] = df["前收盘价"] * 0.8

        # 计算北交所的涨停价和跌停价󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        cond_bj = df["股票代码"].str.contains("bj", regex=False)
        df.loc[cond_bj, "涨停价"] = df["前收盘价"] * 1.3
        df.loc[cond_bj, "跌停价"] = df["前收盘价"] * 0.7

    # 感谢郭毅老板提供的代码，https://bbs.quantclass.cn/thread/55667󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    def price_round(number: float, *, ndigits: int = 2, rounding: str = ROUND_HALF_UP) -> float:
        """对价格进行凑整处理

        北交所规定"超过涨跌幅限制的申报为无效申报"，因此需要对涨跌停价采取截断操作，
        其余市场的涨跌停价及常规价格均采取四舍五入方式凑整。

        Args:
            number (float): 价格（非负数）
            ndigits (int, optional): 价格精度（非负数），默认为2
            rounding (str, optional): 凑整方式，支持如下：
                ROUND_HALF_UP - 默认，四舍五入
                ROUND_UP - 向上取整，用于北交所跌停价计算
                ROUND_DOWN - 向下取整，用于北交所涨停价计算

        Returns:
            float: 凑整后的结果
        """
        return float(
            Decimal(number + (-1e-7 if rounding == ROUND_UP else 1e-7)).quantize(
                Decimal(f"0.{'0' * ndigits}"), rounding
            )
        )

    # 涨跌停价格凑整，北交所截断，其他市场四舍五入󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # apply太慢了，改用np+for循环󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df["涨停价"] = np.where(󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    #     cond_bj, df["涨停价"].apply(lambda x: price_round(x, rounding=ROUND_DOWN)), df["涨停价"].apply(price_round)󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # )󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # df["跌停价"] = np.where(󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    #     cond_bj, df["跌停价"].apply(lambda x: price_round(x, rounding=ROUND_UP)), df["跌停价"].apply(price_round)󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # )󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    zt_price = df["涨停价"].values
    dt_price = df["跌停价"].values
    bj_mask = cond_bj.values

    # 批量处理󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    price_digits = 3 if is_etf else 2
    results = [
        (
            (
                price_round(zt_price[i], ndigits=price_digits, rounding=ROUND_DOWN),
                price_round(dt_price[i], ndigits=price_digits, rounding=ROUND_UP),
            )
            if bj_mask[i]
            else (price_round(zt_price[i], ndigits=price_digits), price_round(dt_price[i], ndigits=price_digits))
        )
        for i in range(len(df))
    ]
    zhang_results, die_results = zip(*results)
    # 一次性赋值󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df["涨停价"] = zhang_results
    df["跌停价"] = die_results

    # 判断是否一字涨停󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df["一字涨停"] = False
    df.loc[df["最低价"] >= df["涨停价"], "一字涨停"] = True

    # 判断是否一字跌停󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df["一字跌停"] = False
    df.loc[df["最高价"] <= df["跌停价"], "一字跌停"] = True

    # 判断是否开盘涨停󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df["开盘涨停"] = False
    df.loc[df["开盘价"] >= df["涨停价"], "开盘涨停"] = True

    # 判断是否开盘跌停󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df["开盘跌停"] = False
    df.loc[df["开盘价"] <= df["跌停价"], "开盘跌停"] = True

    # 判断是否涨停（当天只要涨停过，就为True）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df["是否涨停"] = False
    df.loc[df["最高价"] >= df["涨停价"], "是否涨停"] = True

    return df


def get_trade_date(index_data: pd.DataFrame) -> pd.DataFrame:
    # ==获取交易日历󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    params = {
        "reportName": "RPTA_WEB_ZGXSRL",
        "columns": "ALL",
        "pageSize": 200,
        "sortColumns": "SDATE",
        "sortTypes": -1,
        "callback": f"jQuery1123{random.randint(10000000000000000, 99999999999999999)}_" + str(int(time.time() * 1000)),
        # 动态生成时间戳󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        "_": int(time.time() * 1000),  # 动态生成时间戳󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    }

    response = requests.get(url, params=params)
    content = response.text

    # 去除JSONP封装󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    start = content.find("(") + 1
    end = content.rfind(")")
    json_data = json.loads(content[start:end])

    # ==处理交易日历󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    holiday_df = pd.DataFrame(json_data["result"]["data"])
    holiday_df = holiday_df[holiday_df["MKT"] == "A股"].sort_values("SDATE").reset_index(drop=True)
    holiday_df["SDATE"] = pd.to_datetime(holiday_df["SDATE"])
    holiday_df["EDATE"] = pd.to_datetime(holiday_df["EDATE"])
    # 日历从最小的时间点开始󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    start = holiday_df["SDATE"].min()
    # 截止到最大的时间点的年底󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    end = pd.to_datetime(f"{holiday_df['EDATE'].max().year}-12-31")

    # 创建日历，并标记不交易的日期󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    date_range = pd.DataFrame(pd.date_range(start=start, end=end), columns=["交易日期"])
    for i in holiday_df.index:
        con = (date_range["交易日期"] >= holiday_df.loc[i, "SDATE"]) & (
            date_range["交易日期"] <= holiday_df.loc[i, "EDATE"]
        )
        date_range.loc[con, "交易日期"] = np.nan

    # 正常的周末也是不交易的󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    con = date_range["交易日期"].dt.weekday.isin([5, 6])
    date_range.loc[con, "交易日期"] = np.nan
    date_range = date_range[date_range["交易日期"].notnull()]

    # ==合并交易日󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    trade_date_df = pd.concat([index_data[["交易日期"]], date_range], ignore_index=True)
    trade_date_df = trade_date_df.drop_duplicates(subset="交易日期", keep="first").reset_index(drop=True)
    return trade_date_df


def save_latest_result(conf, select_result_df):
    if not pd.isnull(conf.end_date):
        return

    index_data = conf.read_index_with_trading_date()
    strategy_map = {}
    if conf.strategy is not None:
        strategy_map[conf.strategy.name] = conf.strategy
    if conf.strategy2 is not None:
        strategy_map[conf.strategy2.name] = conf.strategy2
    latest_list = []
    for strategy_name, strategy_df in select_result_df.groupby("策略", observed=True):
        strategy = strategy_map[strategy_name]
        last_date = strategy_df["选股日期"].max()
        calendar_row = index_data[index_data["交易日期"] == last_date]
        if calendar_row.empty or not calendar_row[f"{strategy.hold_period_name}终止日"].iloc[0]:
            continue

        latest_df = strategy_df[strategy_df["选股日期"] == last_date].copy()
        latest_df["交易日期"] = calendar_row["次交易日"].iloc[0]
        latest_list.append(latest_df)

    if not latest_list:
        return

    keep_cols = [
        "选股日期",
        "交易日期",
        "股票代码",
        "股票名称",
        "策略",
        "持仓周期",
        "换仓时间",
        "目标资金占比",
        "选股因子排名",
    ]
    new_select_df = pd.concat(latest_list, ignore_index=True)[keep_cols]
    new_result_path = conf.get_result_folder() / "最新选股结果.csv"
    if new_result_path.exists():
        old_result_df = pd.read_csv(new_result_path, encoding="utf-8-sig", parse_dates=["选股日期", "交易日期"])
        if set(keep_cols).issubset(old_result_df.columns):
            latest_keys = new_select_df[["选股日期", "交易日期", "策略"]].drop_duplicates()
            old_result_df = old_result_df.merge(
                latest_keys.assign(_replace=True), on=["选股日期", "交易日期", "策略"], how="left"
            )
            old_result_df = old_result_df[old_result_df["_replace"].isna()].drop(columns="_replace")
            new_select_df = pd.concat([old_result_df[keep_cols], new_select_df], ignore_index=True)

    new_select_df.to_csv(new_result_path, encoding="utf-8-sig", index=False)


def select_analysis(conf, strategy, select_df: pd.DataFrame, top_n=10, show_plot=True) -> None:
    # 新增函数：分析每年的选股情况󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # 新增：获取所有股票最新的名字󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    last_stock_name = pd.DataFrame(select_df.groupby("股票代码")["股票名称"].last()).reset_index()
    # 每年选股次数n的股票󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    select_df["年份"] = select_df["交易日期"].dt.year
    # 每年的次数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    year_count = (
        pd.DataFrame(select_df.groupby(["年份", "股票代码"])["股票代码"].count())
        .rename(columns={"股票代码": "选中次数"})
        .reset_index()
    )
    # 合并股票名称󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    year_count = year_count.merge(last_stock_name, on="股票代码", how="left")
    # 计算选中次数排名󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    year_count["选中次数_排名"] = year_count.groupby("年份")["选中次数"].rank(method="min", ascending=False)
    year_count = year_count[year_count["选中次数_排名"] <= top_n]
    # 每年选择排名靠前的股票󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    groups = year_count.groupby("年份")
    years = pd.DataFrame()
    for t, g in groups:
        inx = 0 if pd.isnull(years.index.max()) else years.index.max() + 1
        years.loc[inx, "年份"] = str(int(t))
        g = g.sort_values(by="选中次数_排名").reset_index()
        g["历年选股最多"] = g["股票名称"].astype(str) + "_" + g["选中次数"].astype(str) + " "
        txt = g["历年选股最多"].sum()
        years.loc[inx, "历年选股最多"] = txt
    print(years)
    # 单策略使用固定文件名；双策略仍保留策略名前缀，避免两份分析结果互相覆盖。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    year_file_prefix = strategy.name if conf.strategy is not None and conf.strategy2 is not None else ""
    years.to_csv(conf.get_result_folder() / f"{year_file_prefix}历年选股最多.csv", encoding="utf-8-sig")

    # 计算每天的平均市值 & 平均分位数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    period_market_value = select_df.groupby("交易日期").agg({"总市值": "mean", "市值分位": "mean"}).reset_index()
    period_market_value["总市值"] /= 100000000
    # 画图󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    data_dict = {"平均市值": "总市值"}
    ds = period_market_value["总市值"]
    txt = f"均值：{ds.mean():.2f}（亿）  中值：{ds.median():.2f}（亿）  最小值：{ds.min():.2f}（亿）  最大值：{ds.max():.2f}（亿）"
    file_prefix = f"{strategy.name}-" if conf.strategy is not None and conf.strategy2 is not None else ""
    draw_equity_curve_plotly(
        period_market_value,
        data_dict,
        "交易日期",
        desc=txt,
        title="持仓平均市值",
        path=conf.get_result_folder() / f"{file_prefix}平均市值.html",
        show=show_plot,
    )

    data_dict = {"平均市值分位数": "市值分位"}
    ds = period_market_value["市值分位"]
    txt = f"均值：{ds.mean():.2f}  中值：{ds.median():.2f}  最小值：{ds.min():.2f}  最大值：{ds.max():.2f}"
    draw_equity_curve_plotly(
        period_market_value,
        data_dict,
        "交易日期",
        desc=txt,
        title="持仓平均市值分位数",
        path=conf.get_result_folder() / f"{file_prefix}平均市值分位数.html",
        show=show_plot,
    )
