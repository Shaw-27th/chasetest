"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import math
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from core.model.backtest_config import BacktestConfig
from core.market_essentials import cal_fuquan_price, cal_zdt_price, merge_with_index_data
from core.model.type_def import INTRADAY_PRICE_COLS

# ====================================================================================================
# ** 配置与初始化 **
# 设置必要的显示选项及忽略警告，以优化代码输出的阅读体验
# ====================================================================================================
warnings.filterwarnings("ignore")  # 忽略不必要的警告
pd.set_option("expand_frame_repr", False)  # 使数据框在控制台显示不换行
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)

# 定义股票数据所需的列󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
DATA_COLS = [
    "股票代码",
    "股票名称",
    "交易日期",
    "开盘价",
    "最高价",
    "最低价",
    "收盘价",
    "前收盘价",
    "成交量",
    "成交额",
    "流通市值",
    "总市值",
]

ETF_DATA_COLS = [
    "交易日期",
    "基金名称",
    "基金代码",
    "前收盘价",
    "开盘价",
    "最高价",
    "最低价",
    "收盘价",
    "成交量",
    "成交额",
    "复权单位净值",       # "后复权因子",
    "换手率",
    "基金份额合计",
    "场内流通份额",
]


def _read_candle_csv(path: Path, usecols: list, optional_cols: list) -> pd.DataFrame:
    """
    读取日线 CSV。usecols 为必需列，缺失即报错；optional_cols 为可选列，数据中心尚未下发时跳过。

    盘中价格列（0935）随日线数据刚性下发，不区分本次回测用不用；但存量数据文件里可能
    还没有这一列，所以用可调用的 usecols 容忍缺失，再回头校验必需列一个都不能少。
    """
    wanted = {*usecols, *optional_cols}
    df = pd.read_csv(path, encoding="gbk", skiprows=1, parse_dates=["交易日期"], usecols=lambda col: col in wanted)
    missing = [col for col in usecols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少必需列：{missing}")
    return df


def read_stock_data(conf: BacktestConfig, code: str) -> pd.DataFrame:
    return _read_candle_csv(conf.stock_data_path / f"{code}.csv", DATA_COLS, INTRADAY_PRICE_COLS)


def get_target_source_path(conf: BacktestConfig, code: str, code_type: str = "stock") -> Path:
    """按 code_type 返回择时或轮动标的的源文件路径。"""
    if code_type == "stock":
        return conf.stock_data_path / f"{code}.csv"
    if code_type == "index":
        return conf.index_data_path / f"{code}.csv"
    # source_code = f"{code[2:]}.{code[:2].upper()}"
    return conf.etf_data_path / f"{code}.csv"   # return conf.etf_data_path / f"{source_code}.csv"


def get_timing_source_path(conf: BacktestConfig, code: str, code_type: str = "stock") -> Path:
    return get_target_source_path(conf, code, code_type)


def read_etf_data(conf: BacktestConfig, code: str) -> pd.DataFrame:
    """读取单个 ETF，并映射为股票预处理流程使用的通用字段。"""
    source_path = get_timing_source_path(conf, code, code_type="etf")
    df = _read_candle_csv(source_path, ETF_DATA_COLS, INTRADAY_PRICE_COLS)
    df.sort_values("交易日期", inplace=True)
    # ETF 文件可能包含成立期占位行；复权因子为空时即使价格非空也不能参与复权和上市天数计算。󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    df.dropna(subset=["收盘价", "前收盘价", "复权单位净值"], inplace=True)       # df.dropna(subset=["收盘价", "前收盘价", "后复权因子"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if df.empty:
        raise ValueError(f"ETF {code} 不存在有效的日线行情和复权单位净值")    # raise ValueError(f"ETF {code} 不存在有效的日线行情和后复权因子")
    df.rename(columns={"基金代码": "股票代码", "基金名称": "股票名称"}, inplace=True)
    # df["股票代码"] = df["股票代码"].map(lambda value: f"{str(value).split('.')[1].lower()}{str(value).split('.')[0]}")
    actual_codes = df["股票代码"].dropna().unique().tolist()
    if actual_codes != [code]:
        raise ValueError(f"ETF 文件与择时标的 {code} 不匹配：期望 {code}，实际 {actual_codes}")

    # ETF 的市值字段由份额与当日收盘价换算，供标准因子和选股中间表复用。󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    df["流通市值"] = df["场内流通份额"] * df["收盘价"]
    df["总市值"] = df["基金份额合计"] * df["收盘价"]
    df["证券类型"] = "etf"
    return df.drop(columns=["基金份额合计", "场内流通份额"])


def read_index_candle_data(conf: BacktestConfig, code: str) -> pd.DataFrame:
    """把指数数据中心的英文列名转换为框架通用日线结构。"""
    source_path = get_target_source_path(conf, code, code_type="index")
    df = pd.read_csv(source_path, encoding="gbk", parse_dates=["candle_end_time"])
    required = {"candle_end_time", "open", "high", "low", "close", "amount"}
    if missing := required - set(df.columns):
        raise ValueError(f"{source_path} 缺少指数行情字段：{sorted(missing)}")
    df = df.sort_values("candle_end_time").dropna(subset=["close"]).reset_index(drop=True)
    pre_close = df["close"].shift().fillna(df["open"])
    volume = (df["amount"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.DataFrame(
        {
            "股票代码": code,
            "股票名称": code,
            "交易日期": df["candle_end_time"],
            "开盘价": df["open"],
            "最高价": df["high"],
            "最低价": df["low"],
            "收盘价": df["close"],
            "前收盘价": pre_close,
            "成交量": volume,
            "成交额": df["amount"],
            "流通市值": 0.0,
            "总市值": 0.0,
            "证券类型": "index",
        }
    )


def read_target_data(conf: BacktestConfig, code: str, code_type: str) -> pd.DataFrame:
    if code_type == "etf":
        return read_etf_data(conf, code)
    if code_type == "index":
        return read_index_candle_data(conf, code)
    return read_stock_data(conf, code)


def iter_stock_data(conf: BacktestConfig, stock_code_list: list[str]):
    if not stock_code_list:
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        current_code = stock_code_list[0]
        current_future = executor.submit(read_stock_data, conf, current_code)

        for next_code in stock_code_list[1:]:
            next_future = executor.submit(read_stock_data, conf, next_code)
            yield current_code, current_future.result()
            current_code, current_future = next_code, next_future

        yield current_code, current_future.result()


def _prepare_stocks_chunk_worker(args):
    """
    多进程 Worker：负责处理一组股票代码的读取、指标计算与指数对齐。
    """
    code_chunk, conf, index_data = args
    results = []
    for code in code_chunk:
        try:
            df = prepare_data_by_stock(conf, code, index_data, candle_df=None)
            if not df.empty:
                actual_code = df["股票代码"].iloc[0]
                results.append((actual_code, df))
        except Exception as e:
            pass
    return results


def prepare_data(conf: BacktestConfig, boost: bool = True):
    start_time = time.time()  # 记录数据准备开始时间󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿

    # 排除板块󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    stock_hints = []
    for board, name in [("kcb", "科创板"), ("cyb", "创业板"), ("bj", "北交所")]:
        if board in conf.excluded_boards:
            print(f"🗑️ [策略配置] 需要排除`{name}`")
            stock_hints.append(name)
    stock_hint = "、".join(stock_hints) if stock_hints else ""
    if stock_hint:
        stock_hint = f"不包括{stock_hint}"
    else:
        stock_hint = "包括所有板块"

    # 1. 获取代码列表。启用择时后不再扫描全市场。󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    if conf.is_target_mode:
        stock_code_list = sorted(conf.target_code_type_map)
        mode_name = "轮动" if conf.is_rotation_mode else "择时"
        print(f"📂 {mode_name}模式只读取目标标的：{stock_code_list}")
    else:
        stock_code_list = []
        for filename in conf.stock_data_path.glob("*.csv"):
            # 排除隐藏文件󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
            if filename.stem.startswith("."):
                continue
            # 判断是否为北交所股票（代码以 'bj' 开头）󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
            if filename.stem.startswith("bj") and ("bj" in conf.excluded_boards):
                continue
            # 判断是否为科创板股票（代码以 'sh68' 开头）󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
            if filename.stem.startswith("sh68") and ("kcb" in conf.excluded_boards):
                continue
            # 判断是否为科创板股票（代码以 'sz30' 开头）󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
            if filename.stem.startswith("sz30") and ("cyb" in conf.excluded_boards):
                continue
            stock_code_list.append(filename.stem)
        stock_code_list = sorted(set(stock_code_list))
        print(f"📂 读取到股票数量：{len(stock_code_list)}，{stock_hint}")

    # 2. 读取并处理指数数据，确保股票数据与指数数据的时间对齐󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    index_data = conf.read_index_with_trading_date(use_start_date=not conf.is_target_mode)
    all_candle_data_dict = {}  # 用于存储所有股票的K线数据󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿

    if conf.is_target_mode:
        for code in stock_code_list:
            code_type = conf.target_code_type_map[code]
            source_path = get_target_source_path(conf, code, code_type)
            if not source_path.exists():
                raise FileNotFoundError(f"未找到目标标的行情文件：{source_path}")
            df = prepare_data_by_stock(conf, code, index_data)
            if df.empty:
                raise ValueError(f"目标标的 {code} 在当前数据区间内没有有效行情")
            all_candle_data_dict[code] = df
    else:
        n_jobs = int(getattr(conf, "n_jobs", 16))
        if boost and n_jobs > 1 and len(stock_code_list) > 10:
            workers = n_jobs
            print(f"🚀 启动多进程数据预处理（并发进程数: {workers}）...")
            chunk_size = max(10, math.ceil(len(stock_code_list) / (workers * 8)))
            chunks = [
                stock_code_list[i : i + chunk_size]
                for i in range(0, len(stock_code_list), chunk_size)
            ]
            task_args = [(chunk, conf, index_data) for chunk in chunks]

            max_in_flight = workers * 2
            task_iter = iter(task_args)

            with ProcessPoolExecutor(max_workers=workers) as executor:
                pending = set()
                for _ in range(max_in_flight):
                    try:
                        pending.add(executor.submit(_prepare_stocks_chunk_worker, next(task_iter)))
                    except StopIteration:
                        break

                with tqdm(total=len(stock_code_list), desc="预处理数据(多进程)") as pbar:
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            chunk_results = fut.result()
                            for code, df in chunk_results:
                                all_candle_data_dict[code] = df
                            pbar.update(len(chunk_results))
                            try:
                                pending.add(executor.submit(_prepare_stocks_chunk_worker, next(task_iter)))
                            except StopIteration:
                                pass

            # 保持字典键顺序与股票代码列表顺序严格一致
            all_candle_data_dict = {
                code: all_candle_data_dict[code]
                for code in stock_code_list
                if code in all_candle_data_dict
            }
        else:
            if boost:
                stock_data_iter = iter_stock_data(conf, stock_code_list)
            else:
                stock_data_iter = ((code, read_stock_data(conf, code)) for code in stock_code_list)

            for code, candle_df in tqdm(stock_data_iter, desc="预处理数据", total=len(stock_code_list)):
                df = prepare_data_by_stock(conf, code, index_data, candle_df)
                if not df.empty:
                    code = df["股票代码"].iloc[0]
                    all_candle_data_dict[code] = df

    if not all_candle_data_dict:
        raise RuntimeError("没有生成任何股票预处理数据，请检查数据路径、日期范围和板块过滤配置")

    # 3. 缓存预处理后的数据󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    cache_path = conf.get_runtime_folder() / "股票预处理数据.pkl"
    print("💾 保存到缓存文件...", cache_path)
    pd.to_pickle(all_candle_data_dict, cache_path)
    conf.cache_set("candle_df_dict", all_candle_data_dict)
    print(f'📅 行情数据最新交易日期：{max(df["交易日期"].max() for df in all_candle_data_dict.values())}')

    # 4. 准备并缓存pivot透视表数据，用于后续回测󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    print("ℹ️ 准备透视表数据...")
    market_pivot_dict = make_market_pivot(all_candle_data_dict)
    pivot_cache_path = conf.get_runtime_folder() / "全部股票行情pivot.pkl"
    print("💾 保存到缓存文件...", pivot_cache_path)
    pd.to_pickle(market_pivot_dict, pivot_cache_path)
    conf.cache_set("pivot_dict_stock", market_pivot_dict)

    print(f"✅ 数据准备耗时：{time.time() - start_time} 秒\n")


def prepare_data_by_stock(
    conf: BacktestConfig, code: str, index_data: pd.DataFrame, candle_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    对股票数据进行预处理，包括合并指数数据和计算未来交易日状态。

    参数:
    conf (BacktestConfig): 回测配置
    code (str): 股票代码
    index_data (DataFrame): 指数数据
    candle_df (DataFrame, optional): 预先读取的股票日线数据；为 None 时在函数内读取

    返回:
    df (DataFrame): 预处理后的数据
    """
    # 保留独立调用兼容性；批量准备时由后台线程提前读取下一只股票。󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    code_type = conf.target_code_type_map.get(code, "stock")
    is_etf = code_type == "etf"
    is_index = code_type == "index"
    if candle_df is None:
        df = read_target_data(conf, code, code_type)
    else:
        df = candle_df
    pct_change = df["收盘价"] / df["前收盘价"] - 1
    turnover_rate = 0.0 if is_index else (df["换手率"] if is_etf else df["成交额"] / df["流通市值"])
    trading_days = df.index.astype("int") + 1
    avg_price = df["收盘价"] if is_index else df["成交额"] / df["成交量"]

    # 一次性赋值提高性能󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    df = df.assign(涨跌幅=pct_change, 换手率=turnover_rate, 上市至今交易天数=trading_days, 均价=avg_price)

    # 复权价计算及涨跌停价格计算󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    if is_index:
        # 指数不存在除权除息，因子若依赖“xxx_复权”列，直接使用原始价格。󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
        df = df.assign(
            复权因子=df["收盘价"] / df["收盘价"].iat[0],
            收盘价_复权=df["收盘价"],
            开盘价_复权=df["开盘价"],
            最高价_复权=df["最高价"],
            最低价_复权=df["最低价"],
        )
    elif is_etf:
        # 数据源的后复权因子只表示分红等调整，必须乘以真实价格才能得到后复权价格。󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
        # adj_ratio = df["后复权因子"] / df["后复权因子"].iat[0]
        fq_close = df["复权单位净值"]   # fq_close = df["收盘价"] * adj_ratio

        # 框架的“复权因子”用于计算周期收益，语义是从首日开始的累计总回报。󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
        total_return_factor = fq_close / fq_close.iat[0]
        df = df.assign(
            复权因子=total_return_factor,
            收盘价_复权=fq_close,
            开盘价_复权=df["开盘价"] / df["收盘价"] * fq_close,
            最高价_复权=df["最高价"] / df["收盘价"] * fq_close,
            最低价_复权=df["最低价"] / df["收盘价"] * fq_close,
        )
    else:
        df = cal_fuquan_price(df, fuquan_type="后复权")
    df = cal_zdt_price(df, code_type="etf" if (is_etf or is_index) else "stock")
    if is_index:
        df = df.assign(涨停价=np.inf, 跌停价=0.0, 一字涨停=False, 一字跌停=False, 开盘涨停=False, 开盘跌停=False, 是否涨停=False)

    # 合并股票与指数数据，补全停牌日期等信息󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    df = merge_with_index_data(df, index_data.copy(), fill_0_list=["换手率"])

    # 停牌日没有盘中成交价，用当日收盘价填充（此时收盘价已被 merge_with_index_data ffill 补全）󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    for col in INTRADAY_PRICE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df["收盘价"])

    # 股票退市时间小于指数开始时间，就会出现空值󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    if df.empty:
        # 如果出现这种情况，返回空的DataFrame用于后续操作󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
        return pd.DataFrame(columns=[*DATA_COLS, *INTRADAY_PRICE_COLS])

    # 计算开盘买入涨跌幅和未来交易日状态󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    df = df.assign(
        下日_是否交易=df["是否交易"].astype("int8").shift(-1),
        下日_一字涨停=df["一字涨停"].astype("int8").shift(-1),
        下日_开盘涨停=df["开盘涨停"].astype("int8").shift(-1),
        下日_是否涨停=df["是否涨停"].astype("int8").shift(-1),
        下日_是否ST=df["股票名称"].str.contains("ST", regex=False).astype("int8").shift(-1),
        下日_是否S=df["股票名称"].str.contains("S", regex=False).astype("int8").shift(-1),
        下日_是否退市=df["股票名称"].str.contains("退", regex=False).astype("int8").shift(-1),
    )

    # 处理最后一根K线的数据：最后一根K线默认沿用前一日的数据󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    state_cols = ["下日_是否交易", "下日_是否ST", "下日_是否S", "下日_是否退市"]
    df[state_cols] = df[state_cols].ffill()

    return df


def make_market_pivot(market_dict):
    """
    构建市场数据的pivot透视表，便于回测计算。

    参数:
    market_dict (dict): 股票K线数据字典

    返回:
    dict: 包含开盘价、收盘价及前收盘价的透视表数据
    """
    cols = ["交易日期", "股票代码", "开盘价", "收盘价", "前收盘价", "涨停价", "跌停价"]
    # 盘中价格列在数据中心下发后才存在，透视表按实际拿到的列构建󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    sample_df = next(iter(market_dict.values()), None)
    if sample_df is not None:
        cols += [col for col in INTRADAY_PRICE_COLS if col in sample_df.columns]
    df_list = [df[cols].dropna(subset="股票代码") for df in market_dict.values()]
    df_all_market = pd.concat(df_list, ignore_index=True)
    col_names = {
        "开盘价": "open",
        "收盘价": "close",
        "前收盘价": "preclose",
        "涨停价": "zhangting",
        "跌停价": "dieting",
    }
    # 盘中价格列的列名（0935/0940）就是 price_array 里的价格点名，直接沿用不改名󠀂󠁥󠀹󠁡󠀳󠀸󠁥󠁥󠀶󠁢󠀸󠀸󠀵󠁥󠀶󠀸󠀹󠁡󠁣󠀲󠀰󠀳󠀱󠀳󠀵󠀳󠀲󠀳󠀴󠀳󠀹󠀳󠀰󠁿
    return {
        col_names.get(col, col): df_all_market.pivot(values=col, index="交易日期", columns="股票代码")
        for col in cols[2:]
    }
