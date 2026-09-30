# Copyright © 2026 邢不行. All Rights Reserved.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 版权所有 © 2026 邢不行。保留一切权利。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
#󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 专有且保密。本软件非开源软件。仅供个人非商业学习研究使用。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 未经授权不得复制、修改或用于商业用途。详见 LICENSE。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
#󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# Proprietary & Confidential. NOT open-source.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# For personal, non-commercial study and research ONLY.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# See LICENSE for full terms.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# ——‌———‌—‌—‌——‌—‌——‌—‌——‌—‌—‌—‌——‌—‌—‌—‌—‌——‌—‌———‌—‌———‌————‌——󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 邢不行 · 微信: xbx8662󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

"""
report_builder_v2.py — 回测报告 (HTML v1.1) 适配层

对照 equity.py / figure.py 的真实数据列重写:
  equity_df 真实列: 交易日期, 账户可用资金, 持仓市值, 印花税, 券商佣金,
                    总资产, 净值, 手续费, 涨跌幅, 实际杠杆, 净值dd2here,
                    沪深300指数, 中证1000指数(后两列由 perf.py merge 进来)
  select_results 真实列: 选股日期(或交易日期), 股票代码, 目标资金占比,
                         策略, 持仓周期, 换仓时间, [调仓类型], [股票名称]

替换 perf.py 中 show_performance_plot 的方式:

    from report_builder_v2 import build_report_html

    build_report_html(
        conf=conf,
        equity_df=equity_df,
        rtn=rtn,
        year_return=year_return,
        select_results=select_results,
        extra_equities=kwargs.get("extra_equities"),
        pre_timing_equity=kwargs.get("pre_timing_equity"),
        title_prefix=title_prefix,
        template_path=Path(__file__).parent / "templates" / "report_template_v2.html",
        out_path=conf.get_result_folder() / f"{title_prefix}资金曲线.html",
    )
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from core.market_essentials import import_index_data
from core.utils.log_kit import logger


# ---------------------------------------------------------------------------󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 主入口󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# ---------------------------------------------------------------------------󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
def _merge_benchmark_indices(conf, equity_df: pd.DataFrame, title_prefix: str, benchmark_codes: tuple) -> pd.DataFrame:
    """把基准指数(沪深300/中证1000)合并到 equity_df，写入 `<name>指数` 列。"""
    bench_name_map = {"sh000300": "沪深300", "sh000852": "中证1000"}
    is_hourly = "小时" in title_prefix
    base_dir = conf.index_hour_data_path if is_hourly else conf.index_data_path

    eq = equity_df.copy()
    if "交易日期" in eq.columns:
        eq["交易日期"] = pd.to_datetime(eq["交易日期"])
    for code in benchmark_codes:
        name = bench_name_map.get(code, code)
        index_path = base_dir / f"{code}.csv"
        if not index_path.exists():
            logger.warning(f"{name}({code})指数数据不存在，无法添加指数数据")
            continue
        index_df = import_index_data(index_path, [conf.start_date, conf.end_date])
        eq = pd.merge(left=eq, right=index_df[["交易日期", "指数涨跌幅"]], on=["交易日期"], how="left")
        eq[name + "指数"] = (eq["指数涨跌幅"] + 1).cumprod()
        del eq["指数涨跌幅"]
    return eq


def to_report_dict(
    *,
    conf,
    equity_df: pd.DataFrame,
    rtn: pd.DataFrame,
    year_return: pd.DataFrame | None = None,
    select_results: pd.DataFrame | None = None,
    extra_equities: dict | None = None,
    pre_timing_equity: pd.Series | None = None,
    title_prefix: str = "",
    benchmark_codes: tuple = ("sh000300", "sh000852"),
) -> dict:
    equity_df = _merge_benchmark_indices(conf, equity_df.copy(), title_prefix, benchmark_codes)
    eq = equity_df.reset_index(drop=True)
    eq["交易日期"] = pd.to_datetime(eq["交易日期"])

    is_hourly = bool((eq["交易日期"].dt.hour != 0).any() or (eq["交易日期"].dt.minute != 0).any())
    bars_per_day = 4 if is_hourly else 1

    date_fmt = "%Y-%m-%d %H:%M" if is_hourly else "%Y-%m-%d"
    dates_iso = eq["交易日期"].dt.strftime(date_fmt).tolist()
    n = len(eq)
    calendar_days = (eq["交易日期"].iloc[-1] - eq["交易日期"].iloc[0]).total_seconds() / 86400
    if calendar_days > 0 and n > 1:
        # 妥协方案：不改前端模板(report_template_v2.html)时，通过调整 bars_per_year 让旧版前端的󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # bars / bars_per_year 在全区间下等价于按 365 自然日折年。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 这个值会被前端复用于局部区间、夏普等指标，因此这些展示是近似口径。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        bars_per_year = n * 365 / calendar_days
    else:
        bars_per_year = 252 * bars_per_day

    # ---- 净值 ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    strat_eq = eq["净值"].to_numpy(dtype=float)

    # ---- 日收益:优先用现成的"涨跌幅"列 ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if "涨跌幅" in eq.columns:
        daily_ret = eq["涨跌幅"].fillna(0.0).to_numpy(dtype=float)
    else:
        daily_ret = np.concatenate([[0.0], strat_eq[1:] / strat_eq[:-1] - 1])

    # ---- 回撤:优先用 figure.py 上游计算的"净值dd2here" ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if "净值dd2here" in eq.columns:
        drawdown = eq["净值dd2here"].fillna(0.0).to_numpy(dtype=float)
    else:
        drawdown = (strat_eq / np.maximum.accumulate(strat_eq)) - 1.0

    # ---- 仓位堆叠数据(figure.py row=2 的核心信息) ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    leverage = eq["实际杠杆"].fillna(0.0).to_numpy(dtype=float) if "实际杠杆" in eq.columns else np.zeros(n)
    cash_ratio = 1.0 - leverage  # 空仓占比󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # ---- 基准指数 ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    benchmarks = {}
    bench_name_map = {"sh000300": "沪深300", "sh000852": "中证1000"}
    for code in benchmark_codes:
        col = bench_name_map.get(code, code) + "指数"
        if col in eq.columns:
            benchmarks[code] = {
                "name": bench_name_map.get(code, code),
                "equity": eq[col].astype(float).ffill().tolist(),
            }

    # ---- 手续费 / 佣金 / 印花税 ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    commission_total = float(eq["手续费"].sum()) if "手续费" in eq.columns else 0.0
    commission_series = eq["手续费"].astype(float).tolist() if "手续费" in eq.columns else [0.0] * n
    stamp_tax_total = float(eq["印花税"].sum()) if "印花税" in eq.columns else 0.0
    broker_total = float(eq["券商佣金"].sum()) if "券商佣金" in eq.columns else 0.0
    asset_series = eq["总资产"].astype(float).tolist() if "总资产" in eq.columns else None

    timeseries = {
        "dates": dates_iso,
        "equity": [float(x) for x in strat_eq],
        "drawdown": [float(x) for x in drawdown],
        "daily_return": [float(x) for x in daily_ret],
        "leverage": [float(x) for x in leverage],
        "cash_ratio": [float(x) for x in cash_ratio],
        "commission": commission_series,
        "benchmarks": benchmarks,
    }
    if asset_series is not None:
        timeseries["total_asset"] = asset_series
    if pre_timing_equity is not None:
        pre_series = pd.Series(pre_timing_equity).astype(float)
        if isinstance(pre_series.index, pd.DatetimeIndex):
            pre_series.index = pd.to_datetime(pre_series.index)
            pre_series = pre_series[~pre_series.index.duplicated(keep="last")].sort_index()
            report_index = pd.DatetimeIndex(eq["交易日期"])
            aligned_pre = pre_series.reindex(report_index)
            if aligned_pre.isna().any():
                aligned_pre = (
                    pre_series.reindex(pre_series.index.union(report_index).sort_values()).ffill().reindex(report_index)
                )
        else:
            pre_series = pre_series.reset_index(drop=True)
            aligned_pre = pre_series if len(pre_series) == n else pre_series.reindex(range(n)).ffill()

        aligned_pre = aligned_pre.ffill().bfill()
        if len(aligned_pre) == n:
            timeseries["pre_timing_equity"] = aligned_pre.tolist()

    # ---- 持仓 / 调仓 + 换手率推算 ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 在小时级模式下，dates_iso 是 "YYYY-MM-DD HH:MM"，但 select_results 的调仓日是日历日。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 用日历日字符串做 turnover bar 的对齐键，调仓日的换手率落在该日首个 bar 上。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    bar_day_iso = eq["交易日期"].dt.strftime("%Y-%m-%d").tolist()
    holdings, daily_turnover = build_holdings(select_results, dates_iso=dates_iso, bar_day_iso=bar_day_iso)
    timeseries["turnover"] = daily_turnover

    # ---- KPIs ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    kpis = compute_kpis(
        strat_eq,
        daily_ret,
        drawdown,
        daily_turnover,
        commission_total,
        eq["交易日期"],
        bars_per_year=bars_per_year,
        bars_per_day=bars_per_day,
    )

    # ---- 月度收益矩阵 ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    monthly = build_monthly_returns(eq, daily_ret, benchmarks)

    # ---- 子策略 ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    sub_strategies = {}
    if extra_equities:
        n_years = max((eq["交易日期"].iloc[-1] - eq["交易日期"].iloc[0]).days / 365.25, 0.001)
        for name, series in extra_equities.items():
            arr = pd.Series(series).astype(float).ffill().to_numpy()
            sub_dd = (arr / np.maximum.accumulate(arr)) - 1.0
            sub_rets = np.concatenate([[0.0], arr[1:] / arr[:-1] - 1])
            sub_rets_clean = sub_rets[1:][~np.isnan(sub_rets[1:])]
            sharpe = (
                float(sub_rets_clean.mean() / sub_rets_clean.std() * np.sqrt(bars_per_year))
                if len(sub_rets_clean) > 1 and sub_rets_clean.std() > 0
                else 0.0
            )
            sub_strategies[name] = {
                "equity": [float(x) for x in arr],
                "kpis": {
                    "annual_return": float(arr[-1] ** (1 / n_years) - 1) if arr[-1] > 0 else 0.0,
                    "max_drawdown": float(sub_dd.min()),
                    "sharpe": sharpe,
                },
            }

    # ---- 风险指标 (右侧风险表) ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    risk_metrics = build_risk_metrics(daily_ret, kpis, benchmarks, eq, bars_per_year=bars_per_year)

    # ---- meta ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    meta = {
        "run_id": getattr(conf, "run_id", f"run_{datetime.now():%Y%m%d_%H%M%S}"),
        "strategy_name": (title_prefix or "") + conf.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "backtest_range": [dates_iso[0], dates_iso[-1]],
        "initial_capital": float(conf.initial_cash),
        "benchmark_codes": list(benchmark_codes),
        "title_prefix": title_prefix,
        "stamp_tax_total": stamp_tax_total,
        "broker_commission_total": broker_total,
        "is_hourly": is_hourly,
        "bars_per_year": bars_per_year,
        "bars_per_day": bars_per_day,
    }

    # ---- year_return 直接透传(若提供) ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    yr_dict = None
    if year_return is not None and not (isinstance(year_return, pd.DataFrame) and year_return.empty):
        try:
            yr_dict = year_return.to_dict()
        except Exception:
            yr_dict = None

    # ---- rtn 透传(原始策略评价表) ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    rtn_dict = None
    if rtn is not None and not rtn.empty:
        try:
            rtn_dict = {
                "rows": rtn.reset_index().rename(columns={rtn.index.name or "index": "项目"}).to_dict(orient="records")
            }
        except Exception:
            rtn_dict = None

    # ---- 最新选股结果(来自 select_results 文件语义，而非持仓快照) ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    latest_select_results = build_latest_select_results(select_results)

    # 当前框架不包含个股小时择时，保留空字段以兼容 pro 的 V2 模板结构。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    daily_signal_view = None

    strategy_config = []
    if conf.rotation_config is not None:
        rotation_config = conf.rotation_config
        strategy_config.append(
            {
                "name": rotation_config.name,
                "cap_weight": rotation_config.cap_weight,
                "code_list": list(rotation_config.code_list),
                "code_type": rotation_config.code_type,
                "rebalance_time": rotation_config.rebalance_time,
                "hold_period": rotation_config.hold_period,
                "rotation": {
                    "name": rotation_config.rotation.name,
                    "factor_list": [factor.to_tuple() for factor in rotation_config.rotation.factor_list],
                    "max_select_num": rotation_config.rotation.max_select_num,
                    "params": rotation_config.rotation.params,
                },
            }
        )
    elif conf.timing_config is not None:
        timing_config = conf.timing_config
        strategy_config.append(
            {
                "name": timing_config.name,
                "cap_weight": timing_config.cap_weight,
                "code": timing_config.code,
                "code_type": timing_config.code_type,
                "rebalance_time": timing_config.rebalance_time,
                "hold_period": timing_config.hold_period,
                "timing": {
                    "name": timing_config.timing.name,
                    "factor_list": [factor.to_tuple() for factor in timing_config.timing.factor_list],
                    "params": timing_config.timing.params,
                },
            }
        )
    elif conf.strategy is not None:
        strategy_config.append(
            {
                **{key: value for key, value in conf.strategy_raw.items() if key != "funcs"},
                "name": conf.strategy.name,
                "cap_weight": conf.strategy.cap_weight,
            }
        )
    if conf.strategy2 is not None:
        strategy_config.append(
            {
                **{key: value for key, value in conf.strategy2_raw.items() if key != "funcs"},
                "name": conf.strategy2.name,
                "cap_weight": conf.strategy2.cap_weight,
            }
        )

    return {
        "schema_version": "1.1",  # 版本递增󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        "meta": meta,
        "kpis": kpis,
        "timeseries": timeseries,
        "sub_strategies": sub_strategies if sub_strategies else None,
        "monthly_returns": monthly,
        "holdings": holdings,
        "daily_signal_view": daily_signal_view,
        "strategy_config": strategy_config,
        "risk_metrics": risk_metrics,
        "raw": {"rtn": rtn_dict, "year_return": yr_dict, "latest_select_results": latest_select_results},
    }


# ---------------------------------------------------------------------------󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# Helpers󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# ---------------------------------------------------------------------------󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
def compute_kpis(
    strat_eq,
    daily_ret,
    drawdown,
    daily_turnover,
    commission_total,
    dates,
    bars_per_year: int = 252,
    bars_per_day: int = 1,
) -> dict:
    n = len(strat_eq)
    n_years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 0.001)

    total_ret = float(strat_eq[-1] / strat_eq[0] - 1)
    ann_ret = float((strat_eq[-1] / strat_eq[0]) ** (1 / n_years) - 1) if strat_eq[0] > 0 else 0.0
    max_dd = float(np.nanmin(drawdown))
    # 最大回撤发生区间：峰值日期 -> 谷值日期󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    peak_val = float(strat_eq[0])
    peak_idx = 0
    max_dd_start = dates.iloc[0]
    max_dd_end = dates.iloc[0]
    for i, val in enumerate(strat_eq):
        if val > peak_val:
            peak_val = float(val)
            peak_idx = i
        dd = float(val) / peak_val - 1.0 if peak_val > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
            max_dd_start = dates.iloc[peak_idx]
            max_dd_end = dates.iloc[i]

    rets = np.asarray(daily_ret, dtype=float)
    rets = rets[~np.isnan(rets)]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = (
        float(rets.mean() / downside.std() * np.sqrt(bars_per_year)) if len(downside) and downside.std() > 0 else 0.0
    )

    win_rate = float((rets > 0).mean()) if len(rets) else 0.0
    avg_win = float(rets[rets > 0].mean()) if (rets > 0).any() else 0.0
    avg_loss = float(-rets[rets < 0].mean()) if (rets < 0).any() else 0.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # 最长回撤持续天数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    max_run = cur = 0
    for v in drawdown:
        cur = cur + 1 if v < 0 else 0
        if cur > max_run:
            max_run = cur
    max_dd_duration_days = max_run // bars_per_day

    annual_turnover = float(sum(daily_turnover) / n_years) if daily_turnover else 0.0

    return {
        "total_return": total_ret,
        "annual_return": ann_ret,
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": int(max_dd_duration_days),
        "max_drawdown_start_date": str(max_dd_start),
        "max_drawdown_end_date": str(max_dd_end),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": float(ann_ret / abs(max_dd)) if max_dd < 0 else 0.0,
        "win_rate": win_rate,
        "profit_loss_ratio": pl_ratio,
        "annual_turnover": annual_turnover,
        "total_commission": float(commission_total),
        "trading_days": int(n // bars_per_day),
    }


def build_monthly_returns(eq: pd.DataFrame, daily_ret, benchmarks) -> dict:
    df = pd.DataFrame({"date": eq["交易日期"], "ret": daily_ret})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly = df.groupby(["year", "month"])["ret"].apply(lambda x: (1 + x).prod() - 1).unstack()
    years = sorted(monthly.index.tolist())
    matrix = []
    for y in years:
        row = []
        for m in range(1, 13):
            v = monthly.loc[y].get(m)
            row.append(None if pd.isna(v) else float(v))
        matrix.append(row)

    yearly_total = [float((1 + df.loc[df["year"] == y, "ret"]).prod() - 1) for y in years]

    yearly_benchmark: dict[str, list[float]] = {}
    for code, info in benchmarks.items():
        bench_eq = np.asarray(info["equity"], dtype=float)
        bench_ret = np.concatenate([[0.0], bench_eq[1:] / bench_eq[:-1] - 1])
        df["b"] = bench_ret
        yearly_benchmark[code] = [float((1 + df.loc[df["year"] == y, "b"]).prod() - 1) for y in years]
        df.drop(columns="b", inplace=True)

    return {
        "years": [int(y) for y in years],
        "matrix": matrix,
        "yearly_total": yearly_total,
        "yearly_benchmark": yearly_benchmark or None,
    }


def build_risk_metrics(daily_ret, kpis, benchmarks, eq, bars_per_year: int = 252) -> list[dict]:
    rets = np.asarray(daily_ret, dtype=float)
    rets = rets[~np.isnan(rets)]
    ann_vol = float(rets.std() * np.sqrt(bars_per_year)) if len(rets) else 0.0
    downside_vol = float(rets[rets < 0].std() * np.sqrt(bars_per_year)) if (rets < 0).any() else 0.0

    items = [
        {"label": "年化波动率", "value": ann_vol, "format": "percent"},
        {"label": "年化下行波动", "value": downside_vol, "format": "percent"},
        {"label": "夏普比率", "value": kpis["sharpe"], "format": "ratio"},
        {"label": "索提诺比率", "value": kpis["sortino"], "format": "ratio"},
        {"label": "卡玛比率", "value": kpis["calmar"], "format": "ratio"},
        {"label": "日胜率", "value": kpis["win_rate"], "format": "percent"},
        {"label": "盈亏比", "value": kpis["profit_loss_ratio"], "format": "ratio"},
        {"label": "最大回撤持续", "value": kpis["max_drawdown_duration_days"], "format": "days"},
        {"label": "年化换手", "value": kpis["annual_turnover"], "format": "number"},
    ]

    # alpha / beta / IR vs 沪深300(若存在)󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if "sh000300" in benchmarks:
        bench_eq = np.asarray(benchmarks["sh000300"]["equity"], dtype=float)
        bench_ret = np.concatenate([[0.0], bench_eq[1:] / bench_eq[:-1] - 1])
        if len(bench_ret) == len(rets) + 1:
            bench_ret = bench_ret[-len(rets) :]
        if len(bench_ret) == len(rets) and rets.std() > 0:
            cov = float(np.cov(rets, bench_ret)[0, 1])
            beta = cov / float(np.var(bench_ret)) if np.var(bench_ret) > 0 else 0.0
            excess = rets - bench_ret
            ir = float(excess.mean() / excess.std() * np.sqrt(bars_per_year)) if excess.std() > 0 else 0.0
            n_years = max((eq["交易日期"].iloc[-1] - eq["交易日期"].iloc[0]).days / 365.25, 0.001)
            bench_ann = float((bench_eq[-1] / bench_eq[0]) ** (1 / n_years) - 1) if bench_eq[0] > 0 else 0.0
            items.extend(
                [
                    {"label": "Beta(vs沪深300)", "value": float(beta), "format": "number"},
                    {"label": "信息比率", "value": ir, "format": "ratio"},
                    {"label": "年化超额", "value": kpis["annual_return"] - bench_ann, "format": "percent"},
                ]
            )
    return items


def build_holdings(
    select_results: pd.DataFrame | None, dates_iso: list[str], bar_day_iso: list[str] | None = None
) -> tuple[dict, list[float]]:
    """
    返回 (holdings_dict, daily_turnover_list)
    daily_turnover[i] = dates_iso[i] 当日的换手率(权重变化绝对值之和的一半)
    bar_day_iso: 与 dates_iso 等长的「日历日」字符串数组（小时模式下用于把调仓日映射到当日首个 bar）。
    """
    last_date = dates_iso[-1] if dates_iso else ""
    daily_turnover = [0.0] * len(dates_iso)

    if select_results is None or select_results.empty:
        return {
            "latest_snapshot": {"date": last_date, "positions": []},
            "top_stocks_by_year": [],
            "rebalance_events": [],
        }, daily_turnover

    df = select_results.copy()
    if "调仓类型" in df.columns:
        df = df.loc[df["调仓类型"].eq("计划")]

    date_col = "选股日期" if "选股日期" in df.columns else ("交易日期" if "交易日期" in df.columns else None)
    if date_col is None or "股票代码" not in df.columns:
        return {
            "latest_snapshot": {"date": last_date, "positions": []},
            "top_stocks_by_year": [],
            "rebalance_events": [],
        }, daily_turnover

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # ---- 换手率推算 ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 保持原有口径：按整组组合在相邻日期之间比较，避免影响主图和 KPI 的年化换手。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    prev_weights: dict[str, float] = {}
    last_weights: dict[str, float] = {}  # 最后一次调仓的完整持仓权重󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    weight_col = "目标资金占比" if "目标资金占比" in df.columns else None
    name_col = "股票名称" if "股票名称" in df.columns else None
    strategy_col = "策略" if "策略" in df.columns else None
    period_col = "持仓周期" if "持仓周期" in df.columns else None
    rebalance_time_col = "换仓时间" if "换仓时间" in df.columns else None
    last_names: dict[str, str] = {}

    # 用日历日字符串建索引——日级模式下 dates_iso 本身就是日历日；小时模式下用 bar_day_iso。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 同一日历日有多个 bar 时，记录该日首个出现的 bar 索引。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    lookup_keys = bar_day_iso if bar_day_iso is not None else dates_iso
    date_to_idx: dict[str, int] = {}
    for i, k in enumerate(lookup_keys):
        if k not in date_to_idx:
            date_to_idx[k] = i

    for d, grp in df.groupby(date_col):
        d_iso = pd.Timestamp(d).strftime("%Y-%m-%d")
        if weight_col:
            w = grp.groupby("股票代码", observed=True)[weight_col].sum()
            cur_weights = w[w > 1e-9].to_dict()
        else:
            codes = grp["股票代码"].astype(str).unique()
            equal = 1.0 / max(len(codes), 1)
            cur_weights = {c: equal for c in codes}

        if name_col:
            cur_names = grp.set_index("股票代码")[name_col].to_dict()
            last_names.update(cur_names)

        # 换手 = sum(|new - old|) / 2󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        all_codes = set(cur_weights) | set(prev_weights)
        turnover = sum(abs(cur_weights.get(c, 0.0) - prev_weights.get(c, 0.0)) for c in all_codes) / 2.0
        if d_iso in date_to_idx:
            daily_turnover[date_to_idx[d_iso]] = float(turnover)

        prev_weights = cur_weights
        last_weights = cur_weights  # 最后一次循环结束后,这就是当前完整持仓󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # ---- 调仓事件 ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 时间轴需要保留 offset 语义，因此事件按 策略 + 持仓周期 + 换仓时间 独立比较。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    events = []
    event_group_cols = [date_col]
    for col in (strategy_col, period_col, rebalance_time_col):
        if col is not None:
            event_group_cols.append(col)
    prev_group_weights: dict[str, dict[str, float]] = {}
    for keys, grp in df.groupby(event_group_cols, sort=True, observed=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_map = dict(zip(event_group_cols, key_values))
        d_iso = pd.Timestamp(key_map[date_col]).strftime("%Y-%m-%d")
        strategy = "" if strategy_col is None or pd.isna(key_map.get(strategy_col)) else str(key_map.get(strategy_col))
        period = "" if period_col is None or pd.isna(key_map.get(period_col)) else str(key_map.get(period_col))
        rebalance_time = (
            ""
            if rebalance_time_col is None or pd.isna(key_map.get(rebalance_time_col))
            else str(key_map.get(rebalance_time_col))
        )
        group_key = f"{strategy}|{period}|{rebalance_time}"

        if weight_col:
            w = grp.groupby("股票代码", observed=True)[weight_col].sum()
            cur_weights = w[w > 1e-9].to_dict()
        else:
            codes = grp["股票代码"].astype(str).unique()
            equal = 1.0 / max(len(codes), 1)
            cur_weights = {c: equal for c in codes}

        prev_weights_for_group = prev_group_weights.get(group_key, {})
        all_codes = set(cur_weights) | set(prev_weights_for_group)
        turnover = sum(abs(cur_weights.get(c, 0.0) - prev_weights_for_group.get(c, 0.0)) for c in all_codes) / 2.0
        added = [c for c in cur_weights if c not in prev_weights_for_group]
        removed = [c for c in prev_weights_for_group if c not in cur_weights]

        events.append(
            {
                "date": d_iso,
                "type": "计划",
                "strategy": strategy,
                "period": period,
                "offset": period.rsplit("_", 1)[-1] if "_" in period else period,
                "rebalance_time": rebalance_time,
                "group_key": group_key,
                "added": [str(c) for c in added],
                "removed": [str(c) for c in removed],
                "turnover": float(turnover),
            }
        )
        prev_group_weights[group_key] = cur_weights

    # ---- 每年高频股票 ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    df["_year"] = df[date_col].dt.year
    top_by_year = []
    for y, grp in df.groupby("_year"):
        counts = grp["股票代码"].value_counts().head(10)
        stocks = []
        for code, cnt in counts.items():
            name = ""
            if name_col is not None:
                nm = grp.loc[grp["股票代码"] == code, name_col]
                if not nm.empty:
                    name = str(nm.iloc[0])
            stocks.append(
                {
                    "code": str(code),
                    "name": name,
                    "hold_days": int(cnt),
                    "contribution": 0.0,  # 真实贡献需要 P&L 数据,暂留 0󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                }
            )
        top_by_year.append({"year": int(y), "stocks": stocks})

    # ---- 最后一次调仓 = 当前持仓快照(用最后一次的完整持仓权重,不是 added) ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    latest = {"date": last_date, "positions": []}
    if last_weights:
        last_event_date = events[-1]["date"] if events else last_date
        latest["date"] = last_event_date
        positions = []
        for code, w in last_weights.items():
            positions.append(
                {
                    "code": str(code),
                    "name": last_names.get(code, ""),
                    "weight": float(w * 100),  # 转成百分数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    "entry_date": last_event_date,  # 简化:都标最后调仓日󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
                    "unrealized_pnl": 0.0,
                }
            )
        positions.sort(key=lambda x: -x["weight"])
        latest["positions"] = positions

    return {"latest_snapshot": latest, "top_stocks_by_year": top_by_year, "rebalance_events": events}, daily_turnover


def build_latest_select_results(select_results: pd.DataFrame | None) -> list[dict]:
    """
    从原始选股结果中提取“最新一次选股”的完整明细。
    优先使用“选股日期”，缺失时回退“交易日期”。
    """
    if select_results is None or select_results.empty:
        return []

    df = select_results.copy()
    date_col = "选股日期" if "选股日期" in df.columns else ("交易日期" if "交易日期" in df.columns else None)
    if date_col is None:
        return []

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.loc[df[date_col].notna()]
    if df.empty:
        return []

    # 仅保留计划调仓，贴近原选股结果展示语义󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if "调仓类型" in df.columns:
        planned = df.loc[df["调仓类型"].eq("计划")]
        if not planned.empty:
            df = planned

    if "策略" in df.columns:
        # 不同持仓周期的最后选股日可能不同，按策略分别保留最新一期。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        latest_dt = df.groupby("策略", observed=True)[date_col].transform("max")
        latest = df.loc[df[date_col].eq(latest_dt)].copy()
    else:
        latest_dt = df[date_col].max()
        latest = df.loc[df[date_col].eq(latest_dt)].copy()
    if "目标资金占比" in latest.columns:
        latest = latest.loc[pd.to_numeric(latest["目标资金占比"], errors="coerce").fillna(0) > 1e-9]

    # 常用展示字段，缺失则自动跳过󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    cols = [date_col, "股票代码", "股票名称", "目标资金占比", "策略", "持仓周期", "换仓时间", "调仓类型"]
    cols = [c for c in cols if c in latest.columns]
    latest = latest[cols]

    # 排序：策略 + 权重降序 + 代码󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    sort_cols = [c for c in ["策略", "目标资金占比", "股票代码"] if c in latest.columns]
    asc_map = {"策略": True, "目标资金占比": False, "股票代码": True}
    ascending = [asc_map[c] for c in sort_cols]
    if sort_cols:
        latest = latest.sort_values(sort_cols, ascending=ascending)

    rows = latest.to_dict(orient="records")
    for row in rows:
        if date_col in row and pd.notna(row[date_col]):
            row[date_col] = str(pd.Timestamp(row[date_col]))
        if "目标资金占比" in row and row["目标资金占比"] is not None:
            try:
                row["目标资金占比"] = float(row["目标资金占比"])
            except Exception:
                pass
    return rows


# ---------------------------------------------------------------------------󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 注入到 HTML 模板󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# ---------------------------------------------------------------------------󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
def build_report_html(
    *, template_path: str | Path, out_path: str | Path, report: dict | None = None, write_json: bool = True, **kwargs
) -> Path:
    """
    生成最终 HTML 报告。
    传入 report 时直接使用；否则用 kwargs 调 to_report_dict 计算。
    write_json=True 时同时落盘同名 .json，便于排查/复用。
    """
    tpl = Path(template_path).read_text(encoding="utf-8")
    if report is None:
        report = to_report_dict(**kwargs)
    report = _to_json_safe(report)

    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"), default=str)
    payload = payload.replace("</", "<\\/")  # 避免 </script> 注入󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    vendor_dir = Path(template_path).parent / "vendor"
    out = tpl.replace("__ECHARTS_INLINE__", _load_echarts(vendor_dir))
    out = out.replace("__FAVICON_DATA_URI__", _load_favicon_data_uri(vendor_dir))
    out = out.replace("__REPORT_JSON__", payload)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding="utf-8")
    if write_json:
        out_path.with_suffix(".json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    return out_path


_ECHARTS_VERSION = "5.5.0"
# 国内镜像优先，海外 CDN 兜底󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
_ECHARTS_CDN_URLS = (
    f"https://cdn.bootcdn.net/ajax/libs/echarts/{_ECHARTS_VERSION}/echarts.min.js",
    f"https://cdn.staticfile.net/echarts/{_ECHARTS_VERSION}/echarts.min.js",
    f"https://npm.elemecdn.com/echarts@{_ECHARTS_VERSION}/dist/echarts.min.js",
    f"https://cdn.jsdelivr.net/npm/echarts@{_ECHARTS_VERSION}/dist/echarts.min.js",
    f"https://cdnjs.cloudflare.com/ajax/libs/echarts/{_ECHARTS_VERSION}/echarts.min.js",
    f"https://unpkg.com/echarts@{_ECHARTS_VERSION}/dist/echarts.min.js",
)


def _ensure_echarts(vendor_dir: Path) -> Path:
    path = vendor_dir / "echarts.min.js"
    if path.exists() and path.stat().st_size > 100_000:
        return path

    import urllib.request
    import urllib.error

    vendor_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"⏬ 首次使用 v2 报告，正在下载 ECharts {_ECHARTS_VERSION} (~1MB)...")
    last_err = None
    for url in _ECHARTS_CDN_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "select-stock-pro"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
            if len(data) < 100_000:
                raise IOError(f"下载内容异常（仅 {len(data)} 字节）")
            tmp = path.with_suffix(".min.js.part")
            tmp.write_bytes(data)
            tmp.replace(path)
            logger.ok(f"✅ ECharts 已缓存到本地: {path}")
            return path
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            last_err = e
            logger.warning(f"   尝试 {url} 失败: {e}")
            continue

    urls_text = "\n".join(f"     - {u}" for u in _ECHARTS_CDN_URLS)
    raise RuntimeError(
        "❌ 无法下载 ECharts，v2 报告无法生成。\n"
        f"   失败原因: {last_err}\n"
        "   请用浏览器从下列任一链接手动下载，重命名为 echarts.min.js 后放到目标路径：\n"
        f"{urls_text}\n"
        f"     存放路径: {path}\n"
        '   或在 config.py 中设置 report_version = "v1" 切回老版报告（不依赖 ECharts）。'
    )


@lru_cache(maxsize=4)
def _load_echarts(vendor_dir: Path) -> str:
    path = _ensure_echarts(vendor_dir)
    return path.read_text(encoding="utf-8").replace("</", "<\\/")


@lru_cache(maxsize=4)
def _load_favicon_data_uri(vendor_dir: Path) -> str:
    path = vendor_dir / "quantclass.ico"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/x-icon;base64,{b64}"


def _to_json_safe(obj: Any) -> Any:
    """
    递归清洗对象，确保可被 json.dumps 序列化：
    - dict 的 key 转成 str
    - Timestamp/datetime 转为 ISO 字符串
    - numpy/pandas 标量转为 Python 原生类型
    - NaN/NaT 转为 None
    """
    if isinstance(obj, dict):
        return {str(_to_json_safe(k)): _to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(v) for v in obj]

    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)

    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)

    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)

    if isinstance(obj, (np.floating, np.float64, np.float32)):
        v = float(obj)
        return None if np.isnan(v) else v

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if obj is pd.NaT:
        return None

    if isinstance(obj, float) and np.isnan(obj):
        return None

    return obj
