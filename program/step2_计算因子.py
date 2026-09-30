"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import time
import warnings
from typing import Dict

import pandas as pd
from tqdm import tqdm

from config import factor_col_limit
from core.fin_essentials import merge_with_finance_data
from core.model.backtest_config import load_config, BacktestConfig
from core.model.strategy_config import get_col_name
from core.utils.factor_hub import FactorHub

# ====================================================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# ** 配置与初始化 **󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 忽略警告并设定显示选项，以优化代码输出的可读性󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# ====================================================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
warnings.filterwarnings("ignore")
pd.set_option("expand_frame_repr", False)
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)

# fmt: off󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 因子计算之后，需要保存的行情数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
FACTOR_COLS = [
    "交易日期", "股票代码", "股票名称",
    "复权因子", "开盘价", "最高价", "最低价", "收盘价", "成交额", "是否交易", "流通市值", "总市值",
    "下日_是否交易", "下日_开盘涨停", "下日_是否ST", "下日_是否退市", "下日_是否涨停", "下日_一字涨停", "上市至今交易天数",
]

# fmt: on󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿


def cal_strategy_factors(
    conf: BacktestConfig,
    stock_code: str,
    candle_df: pd.DataFrame,
    fin_data: Dict[str, pd.DataFrame] | None = None,
    factor_col_name_list=(),
) -> pd.DataFrame:
    """
    计算指定股票在本次运行中需要的全部因子。

    参数:
    conf (BacktestConfig): 当前回测配置，提供需要计算的因子及参数。
    stock_code (str): 股票代码，仅用于异常信息定位。
    candle_df (DataFrame): 单只股票的日线K线数据。
    fin_data (dict): 财务数据，包括处理后的财务数据和原始财报数据。
    factor_col_name_list: 当前分片需要计算的因子列名；不在列表中的因子会跳过。

    返回:
    DataFrame: 公共日线K线字段和本轮因子列，行数与输入 candle_df 一致。
    """
    factor_series_dict = {}
    before_len = len(candle_df)
    target_columns = set(factor_col_name_list)
    candle_df.sort_values(by="交易日期", inplace=True)

    for factor_name, param_list in conf.factor_params_dict.items():
        factor_file = FactorHub.get_by_name(factor_name)
        for param in param_list:
            col_name = get_col_name(factor_name, param)
            if col_name not in target_columns:
                continue
            factor_df = factor_file.add_factor(candle_df.copy(deep=False), param, fin_data=fin_data, col_name=col_name)

            factor_series_dict[col_name] = factor_df[col_name].values
            # 检查因子计算是否出错󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            if before_len != len(factor_series_dict[col_name]):
                print(f"{stock_code}的{factor_name}因子({param}，{col_name})导致数据长度发生变化，请检查！")
                raise Exception("因子计算出错，请避免在 add_factor 中修改数据行数")

    kline_with_factor_dict = {**{col_name: candle_df[col_name] for col_name in FACTOR_COLS}, **factor_series_dict}
    kline_with_factor_df = pd.DataFrame(kline_with_factor_dict)
    kline_with_factor_df.sort_values(by="交易日期", inplace=True)
    return kline_with_factor_df.reset_index(drop=True)


def process_by_stock(conf: BacktestConfig, stock_code: str, candle_df: pd.DataFrame, factor_col_name_list=()):
    """
    合并单只股票的财务数据并计算当前因子分片。

    普通选股返回公共日线字段和因子列；独立 timing 模式还会保留完整行情字段，
    供择时信号计算继续使用。两个返回结果都保持原始日线顺序和行数。
    """
    candle_df = candle_df.copy()
    # 导入财务数据，将个股数据与财务数据合并，并计算财务指标的衍生指标󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if conf.fin_cols:  # 前面已经完成财务路径预检，这里只处理当前股票的数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 分别为：个股数据、财务数据、原始财务数据（不抛弃废弃的报告数据）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        candle_df, fin_df, raw_fin_df = merge_with_finance_data(conf, stock_code, candle_df)
        fin_data = {"财务数据": fin_df, "原始财务数据": raw_fin_df}
    else:
        fin_data = None

    factor_df = cal_strategy_factors(
        conf, stock_code, candle_df, fin_data=fin_data, factor_col_name_list=factor_col_name_list
    )
    timing_factor_df = None
    if conf.is_target_mode:
        timing_factor_df = candle_df.copy()
        for factor_col_name in factor_col_name_list:
            timing_factor_df[factor_col_name] = factor_df[factor_col_name].values
        timing_factor_df.sort_values("交易日期", inplace=True)

    return factor_df, timing_factor_df


def calculate_factors(conf: BacktestConfig):
    """
    计算所有股票的因子，分为三步：
    1. 加载股票K线数据
    2. 计算每个股票的因子，并存储到列表
    3. 合并所有因子数据并存储

    参数:
    conf (BacktestConfig): 回测配置
    """
    print("🌀 开始计算因子...")
    s_time = time.time()

    # ====================================================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 1. 加载股票K线数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # ====================================================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    print("ℹ️ 配置信息检查...")
    if len(conf.fin_cols) > 0 and not conf.has_fin_data:
        print(f"⚠️ 策略需要财务因子{conf.fin_cols}，但缺少财务数据路径")
        raise ValueError("请在 config.py 中配置财务数据路径")
    elif len(conf.fin_cols) > 0:
        print(f"ℹ️ 检测到财务因子：{conf.fin_cols}")

    print("ℹ️ 读取股票K线数据...")
    candle_df_dict: Dict[str, pd.DataFrame] | None = conf.cache_get("candle_df_dict")
    if candle_df_dict is None:
        candle_df_dict = pd.read_pickle(conf.get_runtime_folder() / "股票预处理数据.pkl")
    else:
        print("📦 命中内存直传缓存 candle_df_dict，跳过磁盘读回")

    factor_col_name_list = conf.factor_col_name_list

    # 择时/轮动复用选股模式的 parquet 分列缓存。目标标的数量有限，一次算完直接落盘。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if conf.is_target_mode:
        timing_factor_df_list = []
        for stock_code, candle_df in tqdm(candle_df_dict.items(), desc="计算因子", total=len(candle_df_dict)):
            _, timing_factor_df = process_by_stock(conf, stock_code, candle_df, factor_col_name_list)
            timing_factor_df_list.append(timing_factor_df)

        print("💾 存储因子数据...")
        timing_factor_df = pd.concat(timing_factor_df_list, ignore_index=True).sort_values(["交易日期", "股票代码"])
        timing_factor_df.reset_index(drop=True, inplace=True)

        # 与普通选股共用同一套缓存结构：all_factors_kline.parquet + factor_<列>.parquet。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 内存缓存仍保留给同进程的 step3/step4；磁盘文件供独立运行 step3 时读回。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        runtime_folder = conf.get_runtime_folder()
        kline_path = runtime_folder / "all_factors_kline.parquet"
        kline_path.unlink(missing_ok=True)
        timing_factor_df[FACTOR_COLS].to_parquet(kline_path, index=False)
        for factor_col_name in factor_col_name_list:
            factor_path = runtime_folder / f"factor_{factor_col_name}.parquet"
            factor_path.unlink(missing_ok=True)
            timing_factor_df[[factor_col_name]].to_parquet(factor_path, index=False)

        # 旧版单个 pickle（择时因子数据.pkl）不再产出，顺手清理历史残留文件󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        (runtime_folder / "择时因子数据.pkl").unlink(missing_ok=True)

        conf.cache_set("target_factor_df", timing_factor_df)
        conf.cache_set("timing_factor_df", timing_factor_df)  # 兼容已有择时调用󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        print(f"✅ 因子计算完成，耗时：{time.time() - s_time:.2f}秒\n")
        return

    # 普通选股按 pro 的日线因子结构分批计算，公共K线与各因子分别缓存。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    runtime_folder = conf.get_runtime_folder()
    all_kline_path = runtime_folder / "all_factors_kline.parquet"
    all_kline_path.unlink(missing_ok=True)
    factor_shards = [
        factor_col_name_list[index : index + factor_col_limit]
        for index in range(0, len(factor_col_name_list), factor_col_limit)
    ] or [[]]

    for shard_index, factor_col_name_shard in enumerate(factor_shards, 1):
        print(f"🗂️ 因子分片计算：{shard_index}/{len(factor_shards)}")
        all_factor_df_list = []
        for stock_code, candle_df in tqdm(candle_df_dict.items(), desc="计算因子", total=len(candle_df_dict)):
            factor_df, _ = process_by_stock(conf, stock_code, candle_df, factor_col_name_shard)
            all_factor_df_list.append(factor_df)

        all_factors_df = pd.concat(all_factor_df_list, ignore_index=True)
        all_factors_df = (
            all_factors_df.assign(
                股票代码=all_factors_df["股票代码"].astype("category"),
                股票名称=all_factors_df["股票名称"].astype("category"),
            )
            .sort_values(by=["交易日期", "股票代码"])
            .reset_index(drop=True)
        )

        print("💾 存储因子数据...")
        if not all_kline_path.exists():
            all_factors_df[FACTOR_COLS].to_parquet(all_kline_path, index=False)

        for factor_col_name in factor_col_name_shard:
            factor_path = runtime_folder / f"factor_{factor_col_name}.parquet"
            factor_path.unlink(missing_ok=True)
            all_factors_df[[factor_col_name]].to_parquet(factor_path, index=False)

    print(f"✅ 因子计算完成，耗时：{time.time() - s_time:.2f}秒\n")


if __name__ == "__main__":
    backtest_config = load_config()
    calculate_factors(backtest_config)
