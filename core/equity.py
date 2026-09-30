"""
邢不行｜策略分享会
股票量化策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

import time
from typing import Dict

import numpy as np
import pandas as pd

import config
from core.evaluate import strategy_evaluate
from core.figure import draw_equity_curve_plotly, save_performance
from core.market_essentials import import_index_data
from core.model.backtest_config import BacktestConfig
from core.model.timing_signal import EquityTiming, INFINITE_HOLD_PERIOD
from core.model.type_def import SimuParams, StockMarketData, get_symbol_type, AdjustRatios
from core.model.type_def import price_array, parse_rebalance_time
from core.rebalance import RebAlways
from core.simulator import Simulator
from core.utils.log_kit import logger

pd.set_option("display.max_rows", 1000)
pd.set_option("expand_frame_repr", False)  # 当列太多时不换行󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿


def show_performance_plot(conf, select_results, equity_df, rtn, year_return, title_prefix="", **kwargs):
    """按配置选择 pro V2 报告或兼容的 Plotly V1 报告。"""
    if config.report_version == "v2":
        from core.perf import show_performance_plot as show_v2

        return show_v2(conf, select_results, equity_df, rtn, year_return, title_prefix=title_prefix, **kwargs)

    account_df = equity_df.copy()
    for index_code, index_name in zip(["sh000300", "sh000905"], ["沪深300", "中证500"]):
        index_path = conf.index_data_path / f"{index_code}.csv"
        if not index_path.exists():
            continue
        index_df = import_index_data(index_path, [account_df["交易日期"].min(), conf.end_date])
        account_df = pd.merge(account_df, index_df[["交易日期", "指数涨跌幅"]], on="交易日期", how="left")
        account_df[f"{index_name}指数"] = (account_df.pop("指数涨跌幅") + 1).cumprod()

    data_dict = {"资金曲线": "净值"}
    for index_name in ["沪深300", "中证500"]:
        col = f"{index_name}指数"
        if col in account_df.columns:
            data_dict[col] = col
    if (pre_timing_equity := kwargs.get("pre_timing_equity")) is not None:
        account_df["再择时前资金曲线"] = pd.Series(pre_timing_equity).reset_index(drop=True)
        data_dict["再择时前资金曲线"] = "再择时前资金曲线"

    draw_equity_curve_plotly(
        account_df,
        data_dict=data_dict,
        date_col="交易日期",
        right_axis={"最大回撤": "净值dd2here"},
        title=(
            f"{conf.name}｜累积净值:{rtn.at['累积净值', 0]}，"
            f"年化收益:{rtn.at['年化收益', 0]}，最大回撤:{rtn.at['最大回撤', 0]}"
        ),
        desc=conf.get_fullname(),
        path=conf.get_result_folder() / f"{title_prefix}资金曲线.html",
    )


def get_stock_market(pivot_dict_stock, trading_dates, symbols, symbol_types) -> StockMarketData:
    df_open: pd.DataFrame = pivot_dict_stock["open"].loc[trading_dates, symbols]
    df_close: pd.DataFrame = pivot_dict_stock["close"].loc[trading_dates, symbols]
    df_preclose: pd.DataFrame = pivot_dict_stock["preclose"].loc[trading_dates, symbols]
    df_dieting: pd.DataFrame = pivot_dict_stock["dieting"].loc[trading_dates, symbols]
    df_zhangting: pd.DataFrame = pivot_dict_stock["zhangting"].loc[trading_dates, symbols]
    # Not sure if necessary󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    should_copy = True

    hour_prices = []
    # PLUMSOFT 于 2025-09-23 优化，可以有效减少内存占用󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    nan_arr = np.full(df_open.shape, np.nan)
    for hour in sorted(price_array):
        if hour in ["open", "close", "preclose", "dieting"]:
            continue
        if hour in pivot_dict_stock.keys():
            hour_prices.append(pivot_dict_stock[hour].loc[trading_dates, symbols].to_numpy(copy=should_copy))
        else:
            hour_prices.append(nan_arr)

    data = StockMarketData(
        candle_begin_ts=(trading_dates.astype(np.int64) // 1000000000).to_numpy(copy=should_copy),
        op=df_open.to_numpy(copy=should_copy),
        cl=df_close.to_numpy(copy=should_copy),
        pre_cl=df_preclose.to_numpy(copy=should_copy),
        dieting=df_dieting.to_numpy(copy=should_copy),
        zhangting=df_zhangting.to_numpy(copy=should_copy),
        types=np.array(symbol_types, dtype=np.int16),
        hour_prices=hour_prices,
    )

    return data


def get_adjust_ratios(df_stock_ratio: pd.DataFrame, start_date, end_date, symbols, reb_time) -> AdjustRatios:
    df_stock_ratio = df_stock_ratio.loc[start_date:end_date, symbols]

    adj_dts = df_stock_ratio.index.to_numpy().astype(np.int64) // 1000000000
    ratios = df_stock_ratio.to_numpy(dtype=np.float64)

    return AdjustRatios(adj_dts=adj_dts, ratios=ratios, reb_time=parse_rebalance_time(reb_time))


def calc_equity(
    conf: BacktestConfig,
    pivot_dict_stock: dict,
    period_ratio_df: dict[tuple, pd.DataFrame],
    symbols: list[str],
    leverage: float | pd.Series = None,
):
    """
    模拟投资组合的表现，生成资金曲线以跟踪组合收益变化。
    :param conf: 回测配置
    :param pivot_dict_stock: 原始数据
    :param period_ratio_df: 目标权重事件流；普通模式按周期生成，timing 模式按信号状态变化生成
    :param symbols: 股票代码
    :param leverage: 杠杆
    :return:
    """
    # 专用模式使用其唯一配置的 code_type。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if conf.rotation_config is not None:
        code_type = conf.rotation_config.code_type
    elif conf.timing_config is not None:
        code_type = conf.timing_config.code_type
    else:
        code_type = "stock"
    symbol_types = [get_symbol_type(sym, code_type=code_type) for sym in symbols]
    # if any(x == BSE_MAIN for x in symbol_types):󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    #     raise ValueError(f'BSE not supported')  # No Beijing stocks󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # 确定回测区间󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    start_date = pd.to_datetime(conf.start_date)
    trading_dates = conf.read_trading_dates(start_date, conf.end_date)
    market_dates = pivot_dict_stock["close"].index
    # 如果交易日期的最新一天 > 行情数据的最新一天 说明是昨天跑的回测，今天又跑了step4󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if trading_dates.iloc[-1] > market_dates[-1]:
        # warning要放在裁剪时间之前，不然trading_dates.iloc[-1]拿到的是裁切后的数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        logger.warning(
            f"交易日期和行情日期不匹配：疑似用之前运行的回测数据，今天再次运行step4。已自动裁剪时间\n"
            f"最新交易日：{trading_dates.iloc[-1]} 最新行情日：{market_dates[-1]}"
        )
        trading_dates = trading_dates.loc[trading_dates <= market_dates[-1]]

    # 读取行情󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    market = get_stock_market(pivot_dict_stock, trading_dates, symbols, symbol_types)

    if leverage is None:
        leverage = 1.0

    if isinstance(leverage, pd.Series):
        leverages = leverage.to_numpy(dtype=np.float64)
    else:
        leverages = np.full(len(market.candle_begin_ts), leverage, dtype=np.float64)

    stay_real = np.full(len(market.candle_begin_ts), int(config.stay_real), dtype=np.int8)

    # 开始回测󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    params = SimuParams(
        init_cash=conf.initial_cash,  # 初始资金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        stamp_tax_rate=0 if code_type in {"etf", "index"} else conf.t_rate,
        commission_rate=conf.c_rate,  # 券商佣金费率󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    )
    logger.debug(
        f"ℹ️ 实际模拟资金:{params.init_cash:,.2f}, "
        f"印花税率:{params.stamp_tax_rate * 100 :.2f}%, "
        f"券商佣金费率:{params.commission_rate * 100 :.2f}%"
    )

    adj_ratios = []
    for keys, df_stock_ratio in period_ratio_df.items():
        # strategy, period, reb_time = keys󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        adj_ratio = get_adjust_ratios(df_stock_ratio, conf.start_date, conf.end_date, symbols, keys[2])
        adj_ratios.append(adj_ratio)

    pos_calc = RebAlways(market.types)

    s_time = time.perf_counter()
    logger.debug("🎯 开始模拟交易...")
    if len(adj_ratios) > 0:
        cashes, pos_values, stamp_taxes, commissions = start_simulation(
            market, params, adj_ratios, leverages, pos_calc, stay_real
        )
    else:
        cashes, pos_values, stamp_taxes, commissions = params.init_cash, 0, 0, 0

    logger.ok(f"完成模拟交易，花费时间: {time.perf_counter() - s_time:.3f}秒")
    account_df = pd.DataFrame(
        {
            "交易日期": trading_dates,
            "账户可用资金": cashes,
            "持仓市值": pos_values,
            "印花税": stamp_taxes,
            "券商佣金": commissions,
        }
    ).reset_index(drop=True)

    account_df["总资产"] = account_df["账户可用资金"] + account_df["持仓市值"]
    account_df["净值"] = account_df["总资产"] / conf.initial_cash

    account_df = account_df.assign(
        手续费=account_df["印花税"] + account_df["券商佣金"],
        涨跌幅=account_df["净值"].pct_change(),
        杠杆=leverages,
        实际杠杆=account_df["持仓市值"] / account_df["总资产"],
    )

    # 策略评价󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    rtn, year_return, month_return, quarter_return = strategy_evaluate(account_df, net_col="净值", pct_col="涨跌幅")
    conf.set_report(rtn.T)

    return account_df, rtn, year_return, month_return, quarter_return


def start_simulation(market, simu_params, adj_ratios, leverages, pos_calc, stay_real):
    """
    模拟股票交易的函数，逐 K 线模拟交易过程，计算账户资金、仓位价值、印花税和佣金等。

    参数:
    - market: StockMarketData 类型，包含市场数据（如 K 线时间戳、价格等）。
    - simu_params: SimuParams 类型，包含模拟参数（如初始资金、佣金率、印花税率等）。
    - adj_ratios: AdjustRatios 类型，包含策略调仓信息（如调仓日期、目标权重、买卖价格索引等）。
    - leverages: np.array 类型，包含动态杠杆
    - pos_calc: 仓位计算函数，用于计算目标买入仓位。
    - stay_real: np.array 类型，表示每根 K 线是否启用真实交易限制。

    返回:
    - cashes: 每根 K 线收盘时的账户可用资金。
    - pos_values: 每根 K 线收盘时的仓位价值。
    - stamp_taxes: 每根 K 线产生的印花税。
    - commissions: 每根 K 线产生的券商佣金。
    """
    # K 线数量󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    n_bars = len(market.candle_begin_ts)

    # 股票品种数量󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    n_syms = len(market.types)

    # 策略数量󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    n_ratios = len(adj_ratios)

    # 账户可用资金 = 初始资金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    available_cash = simu_params.init_cash

    # 记录每根 K 线收盘时的仓位价值󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    pos_values = np.zeros(n_bars, dtype=np.float64)

    # 记录每根 K 线收盘时的账户可用资金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    cashes = np.zeros(n_bars, dtype=np.float64)

    # 记录每根 K 线产生的印花税󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    stamp_taxes = np.zeros(n_bars, dtype=np.float64)

    # 记录每根 K 线产生的券商佣金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    commissions = np.zeros(n_bars, dtype=np.float64)

    # 为每个策略创建模拟器󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    sims = []
    for i in range(n_ratios + 1):
        sim = Simulator(0, simu_params.commission_rate, simu_params.stamp_tax_rate, np.zeros(n_syms, dtype=np.float64))
        sims.append(sim)
    # 跌停模拟器，独立于【策略模拟器】之外，用于处理跌停还卖出成功的BUG󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    dieting_sim = sims[-1]

    """
    下面这些一维/二维变量用于在主循环中记录每个策略在“本根 K 线”下的待执行状态
    与目标权重。它们会在当天不同价格点/收盘后被读取或更新。
    """
    # 策略的调仓周期索引，用于跟踪每个策略的调仓日期。所以没有日期的概念，只有调仓周期的概念󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    adj_dt_idxes = np.zeros(n_ratios, dtype=np.int64)

    # 策略的调仓日期索引：(当日信号，次日信号)󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # - sell_dt_idxes: 卖出调仓日期索引，0 表示不调仓，1 表示调仓。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    sell_dt_idxes = np.full((n_ratios, 2), 0, dtype=np.int8)
    # - buy_dt_idxes: 买入调仓日期索引，0 表示不调仓，1 表示调仓。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    buy_dt_idxes = np.full((n_ratios, 2), 0, dtype=np.int8)

    # 策略的调仓价格索引：󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # - sell_price_idxes: 卖出价格索引，与 market.prices 对应。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    sell_price_idxes = np.zeros(n_ratios, dtype=np.int8)
    # - buy_price_idxes: 买入价格索引，与 market.prices 对应。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    buy_price_idxes = np.zeros(n_ratios, dtype=np.int8)

    # 策略的买入权重矩阵，形状为: 策略数 * 股票品种数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    buy_ratios = np.zeros((n_ratios, n_syms), dtype=np.float64)  # 当前调仓的买入权重󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    next_buy_ratios = np.zeros((n_ratios, n_syms), dtype=np.float64)  # 下一个调仓的买入权重󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # 与 pro 最新真实模式一致：涨停未成交买单只在当日日内等待开板，收盘撤单。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    PENDING_LOTS = 0
    PENDING_CASH = 1
    pending_buy = np.zeros((2, n_ratios, n_syms), dtype=np.float64)
    has_pending_buy = False

    # 主循环：逐 K 线（交易日）模拟整段回测。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 当天流程：󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 1) 开盘前：用前收盘价刷新各策略持仓价格（作为 T+1 的持仓价值基线）。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 2) 设置调仓：若某策略的调仓日等于今天，记录其卖/买的执行“日偏移”(T+0/T+1) 与执行“价格点索引”。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 3) 连续竞价：按价格点顺序执行——先“只卖不买”释放现金，再按目标权重“买/换仓”，并在每个价格点结算持仓价值与最新价。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 4) 收盘：将 T+1 任务的日偏移从 1 递减为 0，并记录当日的资金/仓位/费用。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    for idx_bar in range(n_bars):
        # 初始化本周期印花税和券商佣金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        stamp_tax = commission = 0.0

        # K 线开盘前操作：用前收盘价更新模拟器的持仓价格󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        for sim in sims:
            sim.fill_last_prices(market.pre_cl[idx_bar])

        # 遍历所有策略：若到了该策略的调仓日期，则设置本次卖/买的执行计划󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # - idx_simu：策略索引；adj_dt_idx：该策略下一次调仓日期在 adj_dts 中的下标；adj_info：调仓信息󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        for idx_simu, (adj_dt_idx, adj_info) in enumerate(zip(adj_dt_idxes, adj_ratios)):
            if adj_dt_idx < len(adj_info.adj_dts) and adj_info.adj_dts[adj_dt_idx] == market.candle_begin_ts[idx_bar]:
                # 配置卖出：sp_idx（sell price index）决定卖出的价格点󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                if adj_info.sp_idx < 0:  # 负数表示 T+0 当日卖出（如 -1=当日收盘价）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    sell_dt_idxes[idx_simu, 0] = 1  # 当日卖出󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 负索引从prices数组末尾倒数，如-1对应收盘价󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    sell_price_idxes[idx_simu] = len(market.prices) + adj_info.sp_idx
                else:  # 非负表示 T+1 次日卖出（0=次日开盘，1=次日09:30，…）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    sell_dt_idxes[idx_simu, 1] = 1  # 次日卖出󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    sell_price_idxes[idx_simu] = adj_info.sp_idx

                # 配置买入：bp_idx（buy price index）决定买入的价格点󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                if adj_info.bp_idx < 0:  # 负数表示 T+0 当日买入（如 -1=当日收盘价）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    buy_dt_idxes[idx_simu, 0] = 1  # 当日买入󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 同一个模拟器内，这个暂时是不会发生变化的，因为换仓时间点是固定的󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    buy_price_idxes[idx_simu] = len(market.prices) + adj_info.bp_idx
                    buy_ratios[idx_simu, :] = adj_info.ratios[adj_dt_idx]
                else:  # 非负表示 T+1 次日买入󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    buy_dt_idxes[idx_simu, 1] = 1  # 次日买入󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 同一个模拟器内，这个暂时是不会发生变化的，因为换仓时间点是固定的󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    buy_price_idxes[idx_simu] = adj_info.bp_idx
                    next_buy_ratios[idx_simu, :] = adj_info.ratios[adj_dt_idx]

                # 设置本次调仓的目标权重分配󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                # 当前这次调仓下，各标的的目标资金占比󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

                # 调仓周期索引递增，指向下一个调仓日期󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                adj_dt_idxes[idx_simu] += 1

        # 连续竞价阶段：逐价格点模拟交易。从 open -> 0930 -> 0935 -> ... -> 1300 -> ... -> 1455-> close 逐个价格点模拟交易󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        for idx_price, last_price in enumerate(market.prices):
            # 更新每个模拟器的持仓价值和最新价格󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            for sim in sims:
                sim.settle_pos_values(last_price[idx_bar])
                sim.fill_last_prices(last_price[idx_bar])

            if np.all(np.isnan(last_price[idx_bar])):
                continue

            # ==================跌停模拟器，每天的换仓价都尝试卖出==================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            if stay_real[idx_bar] == 1:
                # 将跌停模拟器可用资金转回账户总可用资金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                dieting_sim_stamp_tax, dieting_sim_commission = dieting_sim.dieting_sell_all(
                    last_price[idx_bar], dieting_sim.is_pos_and_dieting(market.dieting[idx_bar])
                )
                stamp_tax += dieting_sim_stamp_tax
                commission += dieting_sim_commission

                sim_cash = dieting_sim.withdraw_all()
                available_cash += sim_cash
            # ================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

            valid_exec_price = np.logical_not(np.isnan(last_price[idx_bar]))
            valid_zhangting = np.logical_not(np.isnan(market.zhangting[idx_bar]))
            is_zhangting = np.logical_and(valid_zhangting, last_price[idx_bar] >= market.zhangting[idx_bar])
            is_zhangting = np.logical_and(valid_exec_price, is_zhangting)

            # 已挂起的涨停买单在后续价格点开板时回到原策略模拟器成交。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            if stay_real[idx_bar] == 1 and has_pending_buy:
                can_buy_pending = np.logical_and(valid_exec_price, np.logical_not(is_zhangting))
                for idx_simu, sim in enumerate(sims[:-1]):
                    pending_lots = pending_buy[PENDING_LOTS, idx_simu]
                    pending_cash = pending_buy[PENDING_CASH, idx_simu]
                    executable_pending = np.logical_and(pending_lots > 1e-8, can_buy_pending)
                    if np.any(executable_pending):
                        delta_values = np.zeros(n_syms, dtype=np.float64)
                        target_values = sim.pos_values.copy()
                        delta_values[executable_pending] = (
                            pending_lots[executable_pending] * last_price[idx_bar][executable_pending]
                        )
                        target_values[executable_pending] += delta_values[executable_pending]
                        sim.deposit(np.sum(pending_cash[executable_pending]))
                        pending_lots[executable_pending] = 0
                        pending_cash[executable_pending] = 0
                        sim_stamp_tax, sim_commission = sim.adjust_positions(
                            last_price[idx_bar], delta_values, target_values
                        )
                        stamp_tax += sim_stamp_tax
                        commission += sim_commission
                        available_cash += sim.withdraw_all()
                has_pending_buy = np.sum(pending_buy[PENDING_CASH]) > 1e-8

            # 判断需要卖出的策略：当日执行且以当前价格换仓󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            need_sell = np.logical_and(sell_dt_idxes[:, 0] == 1, sell_price_idxes == idx_price)

            # 判断需要买入的策略：当日执行且以当前价格换仓󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            need_buy = np.logical_and(buy_dt_idxes[:, 0] == 1, buy_price_idxes == idx_price)

            # 先处理“只卖不买”的策略：释放现金，避免买入受限  注，最后一个模拟器是跌停模拟器，所以要过滤掉󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            for idx_simu, sim in enumerate(sims[:-1]):
                if need_sell[idx_simu] and not need_buy[idx_simu] and has_pending_buy:
                    pending_cash_for_sim = np.sum(pending_buy[PENDING_CASH, idx_simu])
                    if pending_cash_for_sim > 1e-8:
                        available_cash += pending_cash_for_sim
                        pending_buy[PENDING_CASH, idx_simu, :] = 0
                        pending_buy[PENDING_LOTS, idx_simu, :] = 0
                        has_pending_buy = np.sum(pending_buy[PENDING_CASH]) > 1e-8
                if stay_real[idx_bar] == 1 and (need_sell[idx_simu] or need_buy[idx_simu]):
                    # 获取【有持仓】且【最新价跌停】的股票󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 把要转入到跌停模拟器的金额给保存下来󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 此处只需要考虑纯卖出，没有轧差的情况󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 通过transfer，将sim中跌停的pos_values给减掉，那么sim.sell_all中，得到的delta_values, target_values就是两个0󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 当delta_values为0时，就不会产生手续费󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 当sell_all执行完毕，跌停的股票产生的手续费为0，且pos_values已经通过transfer转移到了跌停模拟器中󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    has_pos_and_dieting = sim.is_pos_and_dieting(market.dieting[idx_bar])
                    # 将资金转入到跌停模拟器中󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    dieting_sim.transfer(sim, has_pos_and_dieting)
                if need_sell[idx_simu] and not need_buy[idx_simu]:
                    # 卖出全部股票，并计算印花税和佣金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    sim_stamp_tax, sim_commission = sim.sell_all(last_price[idx_bar])
                    stamp_tax += sim_stamp_tax
                    commission += sim_commission

                    # 将模拟器可用资金转回账户总可用资金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    sim_cash = sim.withdraw_all()
                    available_cash += sim_cash

            # 计算账户总权益（可用资金 + 所有模拟器的仓位价值），并应用当日杠杆󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            pending_cash_total = np.sum(pending_buy[PENDING_CASH]) if has_pending_buy else 0.0
            total_equity = available_cash + pending_cash_total + sum([sim.get_pos_value() for sim in sims])
            total_equity *= leverages[idx_bar]

            # 再处理需要买入/换仓的策略󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            for idx_simu, (sim, adj_dt_idx, ratios) in enumerate(zip(sims, adj_dt_idxes, buy_ratios)):
                if need_buy[idx_simu]:
                    if has_pending_buy:
                        pending_cash_for_sim = np.sum(pending_buy[PENDING_CASH, idx_simu])
                        if pending_cash_for_sim > 1e-8:
                            available_cash += pending_cash_for_sim
                            pending_buy[PENDING_CASH, idx_simu, :] = 0
                            pending_buy[PENDING_LOTS, idx_simu, :] = 0
                            has_pending_buy = np.sum(pending_buy[PENDING_CASH]) > 1e-8

                    # 目标建仓权益 = 当日总权益 × 该策略本次调仓的资金占比之和󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    ratio_sum = np.sum(ratios)
                    target_equity = total_equity * ratio_sum

                    # 最大可达权益 = 策略仓位价值 + 总可用资金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    max_possible_equity = sim.get_pos_value() + available_cash

                    # 若即使转入全部可用现金也达不到目标，则下调为最大可达权益󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    if max_possible_equity < target_equity:
                        target_equity = max_possible_equity

                    # 需要转入资金 = max(目标建仓权益 - 当前仓位价值, 0)󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    if target_equity > sim.get_pos_value():
                        required_cash = target_equity - sim.get_pos_value()
                    else:
                        required_cash = 0

                    # 将建仓所需资金存入策略模拟器󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    available_cash -= required_cash
                    sim.deposit(required_cash)

                    # 归一化持仓权重（若占比和≈0，则不下单）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    if abs(ratio_sum) < 1e-8:
                        ratios_norm = np.zeros(n_syms, dtype=np.float64)
                    else:
                        ratios_norm = ratios / ratio_sum

                    # 基于目标建仓权益与归一化权重，计算各标的目标持仓󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    target_pos = pos_calc.calc_lots(target_equity, last_price[idx_bar], ratios_norm)

                    # 之前做跌停模拟器的时候把逻辑拆开了，后来发现又不用拆了，暂时先这样󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 注意！目前在处理跌停的时候，不会考虑轧差的情况󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 比如买2手，卖3手，理论上轧差后需要补卖1手。但目前是买2手，卖的3手由于卖不出去，所以会交给跌停模拟器󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    delta_values, target_values = sim.calc_delta_values(last_price[idx_bar], target_pos)

                    frozen_cash = 0.0
                    if stay_real[idx_bar] == 1:
                        limit_up_buy_mask = np.logical_and(delta_values > 0, is_zhangting)
                        if np.any(limit_up_buy_mask):
                            pending_lots = pending_buy[PENDING_LOTS, idx_simu]
                            pending_cash = pending_buy[PENDING_CASH, idx_simu]
                            pending_lots[limit_up_buy_mask] += (
                                delta_values[limit_up_buy_mask] / last_price[idx_bar][limit_up_buy_mask]
                            )
                            frozen_values = delta_values[limit_up_buy_mask] * (1.0 + simu_params.commission_rate)
                            pending_cash[limit_up_buy_mask] += frozen_values
                            frozen_cash = np.sum(frozen_values)
                            target_values[limit_up_buy_mask] = sim.pos_values[limit_up_buy_mask]
                            delta_values[limit_up_buy_mask] = 0
                            has_pending_buy = True

                    # ==================巨坑代码==================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 旧 Numba 参考实现中，对 jitclass 数组属性使用布尔索引原地修改不生效。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 参考实现仍通过 sim 方法修改，保证与迁移前资金曲线一致。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # dieting_sim.pos_values[re_sell] += sim.pos_values[re_sell]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # sim.pos_values[re_sell] -= sim.pos_values[re_sell]󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # ===========================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    # 调整仓位并统计费用󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    sim_stamp_tax, sim_commission = sim.adjust_positions(
                        last_price[idx_bar], delta_values, target_values
                    )

                    commission += sim_commission
                    stamp_tax += sim_stamp_tax

                    # 将模拟器可用资金转回账户总可用资金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    sim_cash = sim.withdraw_all()
                    available_cash += sim_cash
                    if frozen_cash > 1e-8:
                        available_cash -= frozen_cash

        if has_pending_buy:
            available_cash += np.sum(pending_buy[PENDING_CASH])
            pending_buy[PENDING_CASH, :, :] = 0
            pending_buy[PENDING_LOTS, :, :] = 0
            has_pending_buy = False

        # 更新调仓任务的“日偏移”：把 T+1（值为1）的任务在收盘后置为 0，表示“次日将变为当日任务”󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 把 T+1（值为1）的任务在收盘后置为 0，表示“次日将变为当日任务”󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        buy_dt_idxes[:, 0] = buy_dt_idxes[:, 1]
        sell_dt_idxes[:, 0] = sell_dt_idxes[:, 1]
        sell_dt_idxes[:, 1] = 0
        buy_dt_idxes[:, 1] = 0

        # 更新下一个调仓的买入权重󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        buy_ratios = next_buy_ratios
        next_buy_ratios = np.zeros((n_ratios, n_syms), dtype=np.float64)

        # 记录本周期数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        stamp_taxes[idx_bar] = stamp_tax
        commissions[idx_bar] = commission
        pos_values[idx_bar] = sum([sim.get_pos_value() for sim in sims])
        cashes[idx_bar] = available_cash

    return cashes, pos_values, stamp_taxes, commissions


# ====================================================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 动态杠杆再择时模拟󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 1. 生成动态杠杆󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 2. 进行动态杠杆再择时的回测模拟󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 3. 保存结果󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# ====================================================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
def simu_equity_timing(
    conf: BacktestConfig, pivot_dict_stock: dict, period_ratio_df: Dict[tuple, pd.DataFrame], symbols: list[str]
):
    """
    动态杠杆再择时模拟
    :param conf: 回测配置
    :param pivot_dict_stock: 全部行情数据
    :param period_ratio_df: 股票目标资金占比
    :param symbols: 股票代码列表
    :return: 资金曲线，策略收益，年化收益
    """
    logger.info(f"资金曲线再择时，生成动态杠杆")

    # 记录开始时间，用于计算耗时󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    s_time = time.time()

    # 读取资金曲线数据，作为动态杠杆计算的基础󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    account_df = pd.read_csv(conf.get_result_folder() / "资金曲线.csv", index_col=0, encoding="utf-8-sig")

    # 生成动态杠杆，根据资金曲线的权益变化进行杠杆调整󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    equity_signal = conf.re_timing.get_equity_signal(account_df)
    logger.ok(f"完成生成动态杠杆，花费时间： {time.time() - s_time:.3f}秒")

    # 将equity_signals的index设置为交易日期󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    equity_signal.index = pd.to_datetime(account_df["交易日期"])

    # 对组合中的每条策略权重流使用同一资金曲线杠杆，不修改基础模拟使用的原始权重。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    adjusted_period_ratio_df = {}
    for keys, df_stock_ratio in period_ratio_df.items():
        adjusted_period_ratio_df[keys] = df_stock_ratio.mul(equity_signal.reindex(df_stock_ratio.index), axis=0).fillna(
            0
        )

    # 记录时间，用于后续动态杠杆再择时的耗时统计󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    s_time = time.time()
    logger.info(f"开始动态杠杆再择时模拟交易，累计回溯{len(account_df):,} 天...")

    # 进行资金曲线的再择时回测模拟󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # - 使用动态杠杆调整后的持仓计算资金曲线󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # - 使用动态杠杆调整后的股票目标资金占比󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # - 计算回测的总体收益、年度收益、季度收益和月度收益󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    account_df, rtn, year_return, month_return, quarter_return = calc_equity(
        conf, pivot_dict_stock, adjusted_period_ratio_df, symbols
    )

    # 保存回测结果，包括再择时后的资金曲线和收益评价指标󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    save_performance(
        conf,
        资金曲线_再择时=account_df,
        策略评价_再择时=rtn,
        年度账户收益_再择时=year_return,
        季度账户收益_再择时=quarter_return,
        月度账户收益_再择时=month_return,
    )

    logger.ok(f"完成动态杠杆再择时模拟交易，花费时间：{time.time() - s_time:.3f}秒")

    # 返回再择时后的资金曲线和收益结果，用于后续分析或评估󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    return account_df, rtn, year_return


# ================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# step4_实盘模拟.py󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# ================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
def simulate_performance(conf: BacktestConfig, select_results=None, show_plot=True, extra_equities=None):
    """
    模拟投资组合的表现，生成资金曲线以跟踪组合收益变化。

    参数:
    conf (BacktestConfig): 回测配置
    select_results (DataFrame): 选股结果数据
    show_plot (bool): 是否显示回测结果图表
    extra_equities (dict, optional): 额外展示在报告中的资金曲线

    返回:
    DataFrame: 策略评价报告
    """
    # 读取行情透视表。所有策略都复用同一个底层模拟器。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    pivot_dict_stock = conf.cache_get("pivot_dict_stock")
    if pivot_dict_stock is None:
        pivot_dict_stock = pd.read_pickle(conf.get_runtime_folder() / "全部股票行情pivot.pkl")

    s_time = time.time()
    if select_results is None:
        select_results = conf.cache_get("select_results")
    if select_results is None:
        select_results = pd.read_pickle(conf.get_result_folder() / "选股结果.pkl")
    if select_results is None or select_results.empty:
        raise ValueError("选股结果为空，无法生成资金曲线")

    select_results = select_results.copy()
    select_results["选股日期"] = pd.to_datetime(select_results["选股日期"])

    logger.debug("🔀 持仓周期权重聚合...")
    symbols = sorted(select_results["股票代码"].unique())
    period_ratio_df = {}
    for (strategy, period, reb_time), group_df in select_results.groupby(
        ["策略", "持仓周期", "换仓时间"], observed=True
    ):
        period_ratio_df[(strategy, period, reb_time)] = group_df.pivot_table(
            index="选股日期",
            columns="股票代码",
            values="目标资金占比",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )

    min_ratio_date_str = min(ratio_df.index.min() for ratio_df in period_ratio_df.values()).strftime("%Y-%m-%d")
    max_ratio_date_str = max(ratio_df.index.max() for ratio_df in period_ratio_df.values()).strftime("%Y-%m-%d")
    has_infinite_period = any(period == INFINITE_HOLD_PERIOD for _, period, _ in period_ratio_df)
    conf.start_date = max(conf.start_date, min_ratio_date_str) if conf.start_date else min_ratio_date_str
    if not conf.end_date:
        conf.end_date = (
            pivot_dict_stock["close"].index.max().strftime("%Y-%m-%d")
            if has_infinite_period
            else max_ratio_date_str
        )
    logger.debug(
        f"🗓️ 回测模拟区间:{conf.start_date}~{conf.end_date}，"
        f"选股结果区间:{min_ratio_date_str}~{max_ratio_date_str}"
    )

    has_periodic_group = any(period != INFINITE_HOLD_PERIOD for _, period, _ in period_ratio_df)
    period_offset = conf.load_period_offset() if has_periodic_group else None
    for keys, df_stock_ratio in period_ratio_df.items():
        period = keys[1]
        if period == INFINITE_HOLD_PERIOD:
            period_ratio_df[keys] = df_stock_ratio.reindex(columns=symbols, fill_value=0).sort_index()
        else:
            rebalance_dates = pd.concat(
                [period_offset.groupby(period)["交易日期"].last(), pd.Series(df_stock_ratio.index)]
            ).drop_duplicates()
            period_ratio_df[keys] = df_stock_ratio.reindex(
                index=rebalance_dates, columns=symbols, fill_value=0
            ).sort_index()
    logger.debug(f"👌 持仓周期权重聚合完成，耗时：{time.time() - s_time:.3f}秒")

    logger.info(f"开始模拟日线交易...")

    account_df, rtn, year_return, month_return, quarter_return = calc_equity(
        conf, pivot_dict_stock, period_ratio_df, symbols
    )
    save_performance(
        conf,
        资金曲线=account_df,
        策略评价=rtn,
        年度账户收益=year_return,
        季度账户收益=quarter_return,
        月度账户收益=month_return,
    )

    # 检查配置中是否启用了择时信号󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    has_equity_signal = isinstance(conf.re_timing, EquityTiming)
    if has_equity_signal:
        logger.info(f"开始计算资金曲线再择时...")
        # 进行再择时回测，计算动态杠杆后的资金曲线和收益指标󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        account_df2, rtn2, year_return2 = simu_equity_timing(conf, pivot_dict_stock, period_ratio_df, symbols)

        # 可选：绘制再择时的资金曲线图表󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        if show_plot:
            # 绘制再择时后的资金曲线并显示各项收益指标󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            show_performance_plot(
                conf,
                select_results,
                account_df2,
                rtn2,
                year_return2,
                title_prefix="再择时-",
                pre_timing_equity=account_df["净值"],
                extra_equities=extra_equities or {},
            )
    elif show_plot:
        show_performance_plot(conf, select_results, account_df, rtn, year_return, extra_equities=extra_equities or {})

    return conf.report
