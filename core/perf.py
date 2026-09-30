# Copyright © 2026 邢不行. All Rights Reserved.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 版权所有 © 2026 邢不行。保留一切权利。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
#󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 专有且保密。本软件非开源软件。仅供个人非商业学习研究使用。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 未经授权不得复制、修改或用于商业用途。详见 LICENSE。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
#󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# Proprietary & Confidential. NOT open-source.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# For personal, non-commercial study and research ONLY.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# See LICENSE for full terms.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# ——‌————‌————‌—‌—‌—‌————‌—‌———————‌————‌——‌—‌—‌——‌———‌—‌—󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 邢不行 · 微信: xbx8662󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

"""
——‌————‌————‌—‌—‌—‌————‌—‌—————‌—‌—‌—‌—‌——‌——‌—‌———‌—‌—‌—‌—‌—
绩效报告 — 生成回测绩效统计与可视化报告 (v1.1: ECharts 单文件)。
——‌————‌————‌—‌—‌—‌————‌—‌—————‌—‌—‌—‌—‌——‌——‌—‌———‌—‌—‌—‌—‌—
"""

import os
import platform
import webbrowser
from pathlib import Path

from core.model.backtest_config import BacktestConfig
from core.report_builder_v2 import build_report_html, to_report_dict
from core.utils.log_kit import logger


def show_performance_plot(
    conf: BacktestConfig,
    select_results,
    equity_df,
    rtn,
    year_return,
    title_prefix: str = "",
    **kwargs,
):
    """
    生成回测可视化报告 (HTML 单文件，ECharts 5.5)。

    :param conf: 回测配置
    :param select_results: 选股结果 (含 调仓类型/股票代码/股票名称/目标资金占比 等)
    :param equity_df: 回测净值 DataFrame (含 交易日期/净值/涨跌幅/实际杠杆/净值dd2here/手续费 等)
    :param rtn: 策略评价表
    :param year_return: 分年收益率
    :param title_prefix: 标题前缀，用于区分不同 period / 再择时
    :param kwargs:
        - extra_equities: dict[str, pd.Series]，子策略净值
        - pre_timing_equity: pd.Series，再择时前的策略净值
    """
    # ---- 1. 控制台日志 (保留原有摘要输出) ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    logger.debug(
        f"""📈 策略评价 --------------------------------
{rtn}

📊 分年收益率 --------------------------------
{year_return}"""
    )
    logger.debug(f'💰 总手续费: ￥{equity_df["手续费"].sum():,.2f}\n')

    logger.info("开始绘制资金曲线 (新版报告 v2)...")

    # ---- 2. 计算报告数据 (纯 dict, 含基准指数合并, 可落盘 / 单测 / 复用) ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    report = to_report_dict(
        conf=conf,
        equity_df=equity_df,
        rtn=rtn,
        year_return=year_return,
        select_results=select_results,
        extra_equities=kwargs.get("extra_equities"),
        pre_timing_equity=kwargs.get("pre_timing_equity"),
        title_prefix=title_prefix,
    )

    # ---- 3. 渲染 HTML ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    template_path = Path(__file__).parent / "templates" / "report_template_v2.html"
    fig_path = conf.get_result_folder() / f"{title_prefix}资金曲线.html"

    out_path = build_report_html(
        template_path=template_path,
        out_path=fig_path,
        report=report,
    )

    size_kb = out_path.stat().st_size / 1024
    logger.ok(f"📄 报告已生成: {out_path} ({size_kb:,.0f} KB)")

    # ---- 4. 自动打开浏览器 ----󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    _open_in_browser(out_path)


def _open_in_browser(path: Path) -> None:
    abs_path = os.path.abspath(str(path))
    system = platform.system().lower()
    try:
        if system == "darwin":
            os.system(f'open "{abs_path}"')
        elif system == "windows":
            os.system(f'start "" "{abs_path}"')
        else:
            webbrowser.open("file://" + abs_path)
    except Exception as e:
        logger.warning(f"自动打开浏览器失败: {e}")

