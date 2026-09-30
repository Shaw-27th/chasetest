# Copyright © 2026 邢不行. All Rights Reserved.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 版权所有 © 2026 邢不行。保留一切权利。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
#󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 专有且保密。本软件非开源软件。仅供个人非商业学习研究使用。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 未经授权不得复制、修改或用于商业用途。详见 LICENSE。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
#󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# Proprietary & Confidential. NOT open-source.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# For personal, non-commercial study and research ONLY.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# See LICENSE for full terms.󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# —‌—‌———‌——‌———‌—‌——‌—‌———‌——‌———‌———‌——‌—‌—————‌—‌—‌———‌——‌󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 邢不行 · 微信: xbx8662󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

"""
—‌—‌———‌——‌———‌—‌——‌—‌———‌——‌———‌—‌—‌—‌—‌——‌——————‌—‌—‌—‌—‌——‌
类型定义 — 框架通用类型别名与常量。
—‌—‌———‌——‌———‌—‌——‌—‌———‌——‌———‌—‌—‌—‌—‌——‌——————‌—‌—‌—‌—‌——‌
"""

import numpy as np

# 定义股票交易所类型的常量󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 北交所(理应拉黑) bjxxxxxx󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
BSE_MAIN = 0

# 上交所主板 sh60xxxx󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
SSE_MAIN = 1

# 上交所科创板 sh68xxxx󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
SSE_STAR = 2

# 深交所主板 sz00xxxx󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
SZSE_MAIN = 3

# 深交所创业板 sz30xxxx󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
SZSE_CHINEXT = 4

# ETF/指数代理；由专用策略的 code_type 显式指定，不根据交易所代码段推断󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
ETF = 5

# 价格序列数据。换仓可以落在哪些价格点，完全由这个列表决定。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 不铺满高级版那套 5 分钟档位，有两个原因：󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
#   1. 基础版的日线数据只准备到 0940，更细的档位没有数据来源；󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
#   2. 这个列表的长度直接决定 StockMarketData.prices 的第一维，以及模拟主循环每根 K 线󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
#      要遍历的价格点个数。铺满 49 档意味着凭空多出 47 份全 NaN 的价格矩阵。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 顺序即索引顺序：open 必须在首位、close 必须在末位，中间按时间递增。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
price_array = ["open", "0935", "0940", "close"]

# 随日线行情一起下发的盘中价格点，不管本次回测用不用，step1 都会准备好。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
INTRADAY_PRICE_COLS = ["0935", "0940"]

# 当前支持的换仓时间，与高级版口径一致，只开放这三种。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 需要放开新的组合时改这里，parse_rebalance_time 会自动跟着放行。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
SUPPORTED_REBALANCE_TIMES = ("open", "close-open", "0935-0935")


def parse_rebalance_time(reb_time) -> tuple[int, int]:
    """
    将换仓时间解析成 (卖出价格索引, 买入价格索引)，索引对应 price_array 中的位置。

    注意执行时点：返回的索引为负表示 T+0 当日执行，非负表示 T+1 次日执行
    （见 core/equity.py 主循环）。因此 "open" 是次日开盘，不是信号日当天开盘。

    - "open"：次日开盘一次性完成卖出和买入（贴近实盘，框架默认）
    - "close-open"：信号当日收盘卖出，次日开盘买入
    - "0935-0935"：次日 9:35 一次性完成卖出和买入

    配置加载阶段也调用本函数做校验，把换仓时间的配置错误挡在 step4 之前。
    """
    if not isinstance(reb_time, str):
        raise ValueError(f"换仓时间必须是字符串，当前为 {reb_time!r}")

    if reb_time not in SUPPORTED_REBALANCE_TIMES:
        raise ValueError(f"换仓时间 {reb_time!r} 暂不支持，当前可选：{'、'.join(SUPPORTED_REBALANCE_TIMES)}")

    match reb_time:
        case "open":
            return 0, 0
        case "close":
            return -1, -1
        case "close-open":
            return -1, 0
        case _:
            sell_time, buy_time = reb_time.split("-")
            return price_array.index(sell_time), price_array.index(buy_time)


class StockMarketData:
    """
    股票市场数据类，用于存储和管理股票市场的历史数据
    """

    def __init__(self, candle_begin_ts, op, cl, pre_cl, dieting, zhangting, types, hour_prices=()):
        """
        初始化股票市场数据

        :param candle_begin_ts: 交易日零点时间戳数组，单位秒
        :param op: 开盘价数据，二维数组，第一维表示股票，第二维表示时间
        :param cl: 收盘价数据，二维数组，第一维表示股票，第二维表示时间
        :param pre_cl: 前收盘价数据，二维数组，第一维表示股票，第二维表示时间
        :param dieting: 跌停价，二维数组，第一维表示股票，第二维表示时间
        :param zhangting: 涨停价，二维数组，第一维表示股票，第二维表示时间
        :param types: 股票所属交易所类型数组，表示每只股票对应的交易所类型
        :param hour_prices: 日内不同时间点价格数据，包含 3 个二维数组的元组，分别表示不同时间点的价格
        """
        # 交易日零点时间戳󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.candle_begin_ts = candle_begin_ts
        # 原生内核直接消费 (价格点, 交易日, 股票) 的连续数组，避免每次模拟再堆叠 49 份行情。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.prices = np.ascontiguousarray(np.stack([op, *hour_prices, cl], axis=0), dtype=np.float64)

        # 前收盘价数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.pre_cl = pre_cl

        # 跌停价格数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.dieting = dieting

        # 涨停价格数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.zhangting = zhangting

        # 股票所属交易所类型󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.types = types


class SimuParams:
    """
    模拟参数类，用于定义模拟交易中的初始资金、交易佣金和印花税等参数
    """

    # 初始资金，单位人民币元󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    init_cash: float

    # 券商佣金费率，表示每次交易（买入或卖出）时按交易金额收取的佣金比例󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    commission_rate: float

    # 印花税率，表示卖出股票时按卖出金额收取的税率󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    stamp_tax_rate: float

    def __init__(self, init_cash, commission_rate, stamp_tax_rate):
        """
        初始化模拟参数

        :param init_cash: 初始资金，单位人民币元
        :param commission_rate: 券商佣金费率，表示每次交易（买入或卖出）时按交易金额收取的佣金比例
        :param stamp_tax_rate: 印花税率，表示卖出股票时按卖出金额收取的税率
        """
        # 设置初始资金󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.init_cash = init_cash

        # 设置券商佣金费率󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.commission_rate = commission_rate

        # 设置印花税率󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.stamp_tax_rate = stamp_tax_rate


class AdjustRatios:
    """
    调仓参数类，用于定义调仓操作的日期、目标权重以及买卖价格索引
    """

    def __init__(self, adj_dts, ratios, reb_time):
        """
        初始化调仓参数

        :param adj_dts: 调仓日期数组，存储每次调仓的时间戳（单位：秒）
        :param ratios: 目标权重矩阵，二维数组，第一维表示调仓日期，第二维表示每个股票的目标权重
        :param reb_time: 买卖价格索引元组，格式为 (卖出价格索引, 买入价格索引)
        """
        # 设置调仓日期󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.adj_dts = adj_dts

        # 设置目标权重矩阵󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.ratios = ratios

        # 设置卖出价格索引和买入价格索引󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.sp_idx, self.bp_idx = reb_time


def get_symbol_type(symbol: str, code_type: str = "stock") -> int:
    """
    根据股票代码判断其所属的交易所类型

    :param symbol: 股票代码，格式为交易所代码 + 股票编号(例如 sh600000, sz000001, bj430090)
    :param code_type: 标的类型，支持 stock / etf / index；ETF 和指数必须显式传入
    :return: 交易所类型常量（如 BSE_MAIN, SSE_MAIN, SSE_STAR, SZSE_MAIN, SZSE_CHINEXT, ETF）
    :raises ValueError: 如果股票代码不符合已知的交易所代码规则，抛出 ValueError
    """
    if code_type in {"etf", "index"}:
        return ETF

    # 判断是否为北交所股票（代码以 'bj' 开头）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if symbol.startswith("bj"):
        return BSE_MAIN  # 北交所󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # 判断是否为上交所股票（代码以 'sh' 开头）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if symbol.startswith("sh"):
        # 判断是否为科创板股票（代码以 'sh68' 开头）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        if symbol.startswith("sh68"):
            return SSE_STAR  # 科创板󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        else:
            return SSE_MAIN  # 上交所主板󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # 判断是否为深交所股票（代码以 'sz' 开头）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if symbol.startswith("sz"):
        # 判断是否为深交所主板股票（代码以 'sz0' 开头）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        if symbol.startswith("sz0"):
            return SZSE_MAIN  # 深交所主板󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        else:
            return SZSE_CHINEXT  # 深交所创业板󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # 如果股票代码不符合已知规则，抛出 ValueError󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    raise ValueError(f"Unknown stock {symbol}")
