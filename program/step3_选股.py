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
import pandas as pd

from core.model.backtest_config import load_config, BacktestConfig
from core.market_essentials import save_latest_result, select_analysis
from core.figure import draw_equity_curve_plotly

# ====================================================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# ** 配置与初始化 **󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 忽略警告并设定显示选项，以优化代码输出的可读性󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# ====================================================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
warnings.filterwarnings("ignore")
pd.set_option("expand_frame_repr", False)
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)

# 双策略最终选股结果直接采用 Pro 版同名字段，供模拟和报告统一消费。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
BASE_RES_COLS = ["选股日期", "股票代码", "股票名称", "策略", "持仓周期", "换仓时间", "目标资金占比", "选股因子排名"]


def select_stocks(conf: BacktestConfig, show_plot=True):
    """
    生成目标权重。选股模式和择时模式输出同一份「选股结果」，供 step4 统一消费。

    选股流程：
    1. 初始化策略配置
    2. 加载并清洗选股数据
    3. 计算选股因子并进行筛选
    4. 缓存选股结果

    参数:
    conf (BacktestConfig): 回测配置
    show_plot (bool): 是否展示选股结果图表

    返回:
    DataFrame: 选股结果
    """
    s_time = time.time()

    if conf.is_rotation_mode:
        return select_by_rotation(conf)

    # 择时模式的标的已由 code 唯一确定，跳过选股本身，直接把择时信号转成同一份目标权重󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if conf.is_timing_mode:
        return select_by_timing(conf)

    print("🌀 开始选股...")

    runtime_folder = conf.get_runtime_folder()
    kline_path = runtime_folder / "all_factors_kline.parquet"
    if not kline_path.exists():
        raise FileNotFoundError(f"缺少 {kline_path}，请先重新运行 step2")
    kline_df = pd.read_parquet(kline_path)
    period_offset = conf.load_period_offset()

    all_select_result_list = []
    for strategy in (conf.strategy, conf.strategy2):
        if strategy is None:
            continue
        select_result_df, analysis_df = select_stocks_by_strategy(strategy, kline_df, runtime_folder, period_offset)

        if not select_result_df.empty:
            all_select_result_list.append(select_result_df)
            select_analysis(conf, strategy, analysis_df, 10, show_plot=show_plot)

    if all_select_result_list:
        all_select_result_df = pd.concat(all_select_result_list, ignore_index=True)
        all_select_result_df = all_select_result_df.sort_values(by=["选股日期", "持仓周期", "选股因子排名"])[
            BASE_RES_COLS
        ].reset_index(drop=True)
    else:
        all_select_result_df = pd.DataFrame(columns=BASE_RES_COLS)

    # 与 Pro 版保持一致：最终合并结果使用固定文件名，不拼策略名或回测名。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    all_select_result_df.to_pickle(conf.get_result_folder() / "选股结果.pkl")
    all_select_result_df.to_csv(conf.get_result_folder() / "选股结果.csv", encoding="utf-8-sig", index=False)
    conf.cache_set("select_results", all_select_result_df)

    if not all_select_result_df.empty:
        save_latest_result(conf, all_select_result_df)

    print(f"💾 合并选股结果数据大小：{all_select_result_df.memory_usage(deep=True).sum() / 1024 / 1024:.4f} MB")
    print(f"✅ 选股完成，总耗时：{time.time() - s_time:.3f}秒\n")
    return all_select_result_df


def select_by_rotation(conf: BacktestConfig):
    """调用仓位管理插件，并把日期×标的权重矩阵转换为调仓事件。"""
    s_time = time.time()
    print(f"🌀 开始轮动：{conf.get_fullname()}")
    stock_df = conf.cache_get("target_factor_df")
    if stock_df is None:
        stock_df = conf.cache_get("timing_factor_df")
    if stock_df is None:
        runtime_folder = conf.get_runtime_folder()
        kline_path = runtime_folder / "all_factors_kline.parquet"
        if not kline_path.exists():
            raise FileNotFoundError(f"缺少 {kline_path}，请先运行 step2")
        stock_df = pd.read_parquet(kline_path)
        load_factor_columns(runtime_folder, stock_df, conf.factor_col_name_list)

    expected_codes = set(conf.target_code_type_map)
    actual_codes = set(stock_df["股票代码"].dropna().unique())
    if actual_codes != expected_codes:
        raise ValueError(f"轮动数据必须且只能包含 {sorted(expected_codes)}，实际为 {sorted(actual_codes)}")
    if stock_df.duplicated(["交易日期", "股票代码"]).any():
        raise ValueError("轮动行情存在重复的交易日期与标的组合")

    rotation_config = conf.rotation_config
    rotation_df = stock_df[stock_df["股票代码"].isin(rotation_config.code_list)].copy()
    rotation_df.sort_values(["交易日期", "股票代码"], inplace=True)
    assets = []
    for code in rotation_config.code_list:
        code_df = rotation_df[rotation_df["股票代码"] == code].copy()
        if code_df.empty:
            raise ValueError(f"轮动标的 {code} 没有可用行情")
        if code_df["交易日期"].duplicated().any():
            raise ValueError(f"轮动标的 {code} 存在重复交易日期")
        missing_columns = {"交易日期", *rotation_config.factor_columns} - set(code_df.columns)
        if missing_columns:
            raise KeyError(f"轮动标的 {code} 缺少字段：{sorted(missing_columns)}")
        code_df.sort_values("交易日期", inplace=True)
        code_df.reset_index(drop=True, inplace=True)
        assets.append({"name": code, "df": code_df})

    ratio = rotation_config.get_ratio(assets)
    if conf.start_date:
        ratio = ratio.loc[ratio.index >= pd.to_datetime(conf.start_date)]
    if conf.end_date:
        ratio = ratio.loc[ratio.index <= pd.to_datetime(conf.end_date)]
    if ratio.empty:
        raise ValueError(f"{rotation_config.name} 在回测区间内没有可用交易日")

    target_ratio = ratio * rotation_config.cap_weight
    adjust_mask = target_ratio.ne(target_ratio.shift()).any(axis=1)
    adjusted_ratio = target_ratio.loc[adjust_mask]
    ratio_df = target_ratio.copy()
    ratio_df.insert(0, "策略", rotation_config.name)
    ratio_df = ratio_df.reset_index(names="交易日期")
    ratio_df.to_csv(conf.get_result_folder() / "轮动仓位.csv", index=False, encoding="utf-8-sig")

    select_result_df = adjusted_ratio.rename_axis("选股日期").reset_index().melt(
        id_vars="选股日期",
        var_name="股票代码",
        value_name="目标资金占比",
    )
    code_names = rotation_df.drop_duplicates("股票代码", keep="last").set_index("股票代码")["股票名称"]
    select_result_df["股票名称"] = select_result_df["股票代码"].map(code_names).fillna(
        select_result_df["股票代码"]
    )
    select_result_df["选股因子排名"] = select_result_df.groupby("选股日期")["目标资金占比"].rank(
        method="first", ascending=False
    )
    select_result_df.loc[select_result_df["目标资金占比"].eq(0), "选股因子排名"] = 0
    select_result_df = select_result_df.assign(
        策略=rotation_config.name,
        持仓周期=rotation_config.hold_period,
        换仓时间=rotation_config.rebalance_time,
    )[BASE_RES_COLS].reset_index(drop=True)
    select_result_df.to_pickle(conf.get_result_folder() / "选股结果.pkl")
    select_result_df.to_csv(conf.get_result_folder() / "选股结果.csv", encoding="utf-8-sig", index=False)
    conf.cache_set("select_results", select_result_df)

    print(
        f"📈 {rotation_config.name}：{ratio.index.min():%Y-%m-%d}~{ratio.index.max():%Y-%m-%d}，"
        f"调仓{int(adjust_mask.sum())}次，空仓{int(target_ratio.sum(axis=1).eq(0).sum())}日"
    )
    print(f"✅ 轮动完成，总耗时：{time.time() - s_time:.3f}秒\n")
    return select_result_df


def select_by_timing(conf: BacktestConfig):
    """
    择时模式的"选股"：标的已由 code 确定，只需把各择时策略的 0/1 持仓状态转成与选股结果同构的
    调仓事件流。

    单个择时策略根据目标标的计算信号，目标资金占比 = cap_weight × 持仓状态，最终进入与选股模式
    完全相同的模拟器。

    返回:
    DataFrame: 与选股模式完全相同的 BASE_RES_COLS 结构
    """
    s_time = time.time()
    print(f"🌀 开始择时：{conf.get_fullname()}")

    stock_df = conf.cache_get("target_factor_df")
    if stock_df is None:
        stock_df = conf.cache_get("timing_factor_df")
    if stock_df is None:
        # 与选股模式共用同一套 parquet 分列缓存，独立运行 step3 时从磁盘读回。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        runtime_folder = conf.get_runtime_folder()
        kline_path = runtime_folder / "all_factors_kline.parquet"
        if not kline_path.exists():
            raise FileNotFoundError(f"缺少 {kline_path}，请先运行 step2")
        stock_df = pd.read_parquet(kline_path)
        load_factor_columns(
            runtime_folder,
            stock_df,
            conf.timing_config.factor_columns,
        )

    actual_codes = set(stock_df["股票代码"].dropna().unique())
    expected_codes = {conf.timing_config.code}
    if actual_codes != expected_codes:
        raise ValueError(f"择时数据必须且只能包含目标标的 {sorted(expected_codes)}，实际为 {sorted(actual_codes)}")

    timing_config = conf.timing_config
    code_df = stock_df[stock_df["股票代码"] == timing_config.code].copy()
    code_df = code_df.sort_values("交易日期").reset_index(drop=True)
    if code_df["交易日期"].duplicated().any():
        raise ValueError(f"择时标的 {timing_config.code} 的行情存在重复交易日期")

    # 信号在完整历史上计算，保证 EMA 一类指标有足够预热，再按回测区间裁剪󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    holding = timing_config.get_timing_signal(code_df)

    in_range = pd.Series(True, index=code_df.index)
    if conf.start_date:
        in_range &= code_df["交易日期"] >= pd.to_datetime(conf.start_date)
    if conf.end_date:
        in_range &= code_df["交易日期"] <= pd.to_datetime(conf.end_date)
    if not in_range.any():
        raise ValueError(f"择时标的 {timing_config.code} 在回测区间内没有可用交易日")

    range_df = code_df.loc[in_range].copy()
    range_holding = holding.loc[in_range]
    # shift 让区间首日必定成为一次调仓，把区间开始时已经持有的仓位显式带进模拟󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    adjust_mask = range_holding.ne(range_holding.shift())
    target_ratio = range_holding * timing_config.cap_weight

    signal_df = range_df[["交易日期", "股票代码", "股票名称", *timing_config.factor_columns]].copy()
    signal_df["策略"] = timing_config.name
    signal_df["持仓状态"] = range_holding.to_numpy()
    signal_df["是否调仓"] = adjust_mask.astype("int8").to_numpy()
    signal_df["目标资金占比"] = target_ratio.to_numpy()
    signal_df.to_csv(conf.get_result_folder() / "择时信号.csv", index=False, encoding="utf-8-sig")

    select_result_df = range_df.loc[adjust_mask, ["交易日期", "股票代码", "股票名称"]].copy()
    select_result_df["目标资金占比"] = target_ratio.loc[adjust_mask].to_numpy()
    select_result_df["选股因子排名"] = 1  # 单标的没有排名，占位保持结构一致󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    select_result_df = (
        select_result_df.assign(
            策略=timing_config.name, 持仓周期=timing_config.hold_period, 换仓时间=timing_config.rebalance_time
        )
        .rename(columns={"交易日期": "选股日期"})[BASE_RES_COLS]
        .reset_index(drop=True)
    )
    select_result_df.to_pickle(conf.get_result_folder() / "选股结果.pkl")
    select_result_df.to_csv(conf.get_result_folder() / "选股结果.csv", encoding="utf-8-sig", index=False)
    conf.cache_set("select_results", select_result_df)

    print(
        f"📈 {timing_config.name}：{range_df['交易日期'].min():%Y-%m-%d}~{range_df['交易日期'].max():%Y-%m-%d}，"
        f"持仓{int(range_holding.sum())}/{len(range_df)}个交易日，调仓{int(adjust_mask.sum())}次"
        f"（其中买入{int((target_ratio > 0).sum())}次）"
    )
    print(f"✅ 择时完成，总耗时：{time.time() - s_time:.3f}秒\n")
    return select_result_df


def load_factor_columns(runtime_folder, kline_df, factor_columns):
    """从 factor_<列>.parquet 读取因子列并入日线K线，行数与K线严格一致。"""
    for factor_col_name in factor_columns:
        factor_path = runtime_folder / f"factor_{factor_col_name}.parquet"
        if not factor_path.exists():
            raise FileNotFoundError(f"缺少 {factor_path}，请先使用当前配置重新运行 step2")
        factor_series = pd.read_parquet(factor_path)[factor_col_name]
        if len(factor_series) != len(kline_df):
            raise ValueError(f"因子缓存与K线行数不一致：{factor_col_name}={len(factor_series)}，K线={len(kline_df)}")
        kline_df[factor_col_name] = factor_series.to_numpy()
    return kline_df


def select_stocks_by_strategy(strategy, kline_df, runtime_folder, period_offset):
    """从共享日线因子面板中完成一个策略的选股，并生成组合模拟权重。"""
    s_time = time.time()
    print(f"[{strategy.name}] 选股策略启动...")

    period_prefix = strategy.hold_period[:-1] if strategy.hold_period.endswith("D") else strategy.hold_period
    period_name = f"{period_prefix}_0"
    if period_name not in period_offset.columns:
        raise KeyError(f"period_offset.csv 缺少持仓周期列：{period_name}")
    select_dates = period_offset.groupby(period_name)["交易日期"].last()
    if select_dates.empty:
        raise ValueError(f"持仓周期 {period_name} 没有可用选股日期")

    period_df = kline_df.copy()
    factor_columns = strategy.factor_columns
    load_factor_columns(runtime_folder, period_df, factor_columns)

    period_df = period_df[
        period_df["交易日期"].between(select_dates.min(), select_dates.max(), inclusive="both")
    ].copy()

    period_df["市值分位"] = period_df.groupby("交易日期")["总市值"].rank(pct=True)
    period_df = period_df[period_df["是否交易"] == 1].dropna(subset=factor_columns).copy()
    period_df.dropna(subset=["股票代码"], inplace=True)
    period_df.sort_values(by=["交易日期", "股票代码"], inplace=True)
    period_df.reset_index(drop=True, inplace=True)
    print(f"[{strategy.name}] 选股数据准备完成，耗时：{time.time() - s_time:.2f}s")

    s = time.time()
    period_df = strategy.filter_before_select(period_df)
    print(f"[{strategy.name}] 前置筛选耗时：{time.time() - s:.2f}s")

    s = time.time()
    result_df = strategy.calc_select_factor(period_df)
    period_df = period_df.join(result_df)
    print(f"[{strategy.name}] 因子计算耗时：{time.time() - s:.2f}s")

    s = time.time()
    period_df = period_df[period_df["交易日期"].isin(select_dates)]
    period_df = select_by_factor(period_df, strategy.select_num, strategy.factor_name)
    period_df = strategy.filter_after_select(period_df).copy()
    if not period_df.empty:
        # 后置过滤可能改变股票数量，因此重新计算策略内等权，再乘组合资金权重。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        period_df["目标资金占比"] = (
            1 / period_df.groupby("交易日期")["股票代码"].transform("size") * strategy.cap_weight
        )
    print(f"[{strategy.name}] 选股耗时：{time.time() - s:.2f}s")

    analysis_df = period_df.copy()
    select_result_df = period_df[["交易日期", "股票代码", "股票名称", "目标资金占比", "选股因子排名"]].copy()
    select_result_df = select_result_df.assign(
        策略=strategy.name, 持仓周期=period_name, 换仓时间=strategy.rebalance_time
    ).rename(columns={"交易日期": "选股日期"})
    return select_result_df[BASE_RES_COLS], analysis_df


def select_by_factor(period_df, select_num: float | int, factor_name):
    """
    基于因子选择目标股票并计算资金权重。

    参数:
    period_df (DataFrame): 筛选后的数据
    select_num (float | int): 选股数量或比例
    factor_name (str): 选股因子名称

    返回:
    DataFrame: 带目标资金占比的选股结果
    """
    period_df = calc_select_factor_rank(period_df, factor_column=factor_name, ascending=True)

    # 基于排名筛选股票󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if int(select_num) == 0:  # 选股数量是百分比󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        period_df = period_df[period_df["选股因子排名"] <= period_df["总股数"] * select_num].copy()
    else:  # 选股数量是固定的数字󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        period_df = period_df[period_df["选股因子排名"] <= select_num].copy()

    # 根据选股数量分配目标资金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    period_df["目标资金占比"] = 1 / period_df.groupby("交易日期")["股票代码"].transform("size")

    period_df.sort_values(by="交易日期", inplace=True)
    period_df.reset_index(drop=True, inplace=True)

    # 清理无关列󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    period_df.drop(columns=["总股数"], inplace=True)

    return period_df


def calc_select_factor_rank(df, factor_column="因子", ascending=True):
    """
    计算因子排名。

    参数:
    df (DataFrame): 原始数据
    factor_column (str): 因子列名
    ascending (bool): 排序顺序，True为升序

    返回:
    DataFrame: 包含排名的原数据
    """
    # 计算因子的分组排名󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df["选股因子排名"] = df.groupby("交易日期")[factor_column].rank(method="min", ascending=ascending)
    # 根据时间和因子排名排序󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df.sort_values(by=["交易日期", "选股因子排名"], inplace=True)
    # 重新计算一下总股数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df["总股数"] = df.groupby("交易日期")["股票代码"].transform("size")
    return df


if __name__ == "__main__":
    backtest_config = load_config()
    select_stocks(backtest_config)
