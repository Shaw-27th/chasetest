"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

from dataclasses import dataclass, field, fields
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd

from core.model.strategy_config import HashableDict, parse_param, FactorConfig
from core.model.type_def import parse_rebalance_time
from core.utils.signal_hub import get_signal_by_name

# 择时不存在固定持仓周期，持有到信号翻转为止。该值会写进选股结果的「持仓周期」列，󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
# 供后续模拟阶段区分"按周期换仓"和"按事件换仓"两种权重流。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
INFINITE_HOLD_PERIOD = "infinite"


@dataclass
class TimingSignal:
    """
    信号库中的一个择时信号，由配置里的 timing 字段解析而来，只负责从指标列判买卖点。

    - name / factor_list / params 对应 timing.name / timing.factor_list / timing.params
    - funcs 是从信号库加载的策略函数（timing_signal 等）
    - 信号函数通过调用方传入的 TimingConfig 自行读取 factor_list 与 params，
      框架不再为信号解析因子列映射（signal 依赖什么因子由信号函数自己声明）
    """

    # 信号名称（timing.name，信号库中的择时逻辑）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    name: str = ""
    # 择时依赖的指标因子（timing.factor_list，四元组格式与选股 factor_list 一致，权重自动归一化）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    factor_list: Tuple[FactorConfig] = field(default_factory=tuple)
    # 信号参数（timing.params，支持 tuple/float/int/str，dict 会被转为可哈希的 HashableDict；󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # 默认值由信号函数自行处理）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    params: Union[tuple, HashableDict, str, int, float, bool, None] = ()

    # 策略函数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    funcs: Dict[str, Callable] = field(default_factory=dict)

    @classmethod
    def init(
        cls, name: str = "", factor_list: List = None, params: Union[tuple, float, int, str] = ()
    ) -> "TimingSignal":
        """从信号库加载择时逻辑：校验参数。参数写错在配置加载阶段就报出来。"""
        name = str(name).strip()
        if not name:
            raise ValueError("timing.name 必须是非空字符串")
        # param的类型需要转换为hashable的状态（list→tuple，dict→HashableDict）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        params = parse_param(params)

        # 择时依赖的指标与选股因子同格式（四元组），权重自动归一化；择时信号不依赖权重，占位即可󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        factor_list = FactorConfig.parse_list(factor_list or [])

        funcs = get_signal_by_name(name)
        if "timing_signal" not in funcs:
            raise ValueError(f"信号库/{name}.py 必须实现 timing_signal(timing_config, stock_df) 函数")

        return cls(name=name, factor_list=tuple(factor_list), params=params, funcs=funcs)

    def __repr__(self) -> str:
        return f"{self.name}{self.params}"


@dataclass
class EquityTiming:
    name: str
    params: list | tuple = ()

    # 策略函数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    funcs: Dict[str, Callable] = field(default_factory=dict)

    @classmethod
    def init(cls, **config) -> "EquityTiming":
        config["params"] = parse_param(config.get("params", ()))
        config["funcs"] = get_signal_by_name(config["name"])
        leverage_signal = cls(**config)

        return leverage_signal

    def get_equity_signal(self, equity_df: pd.DataFrame) -> pd.Series:
        return self.funcs["equity_signal"](equity_df, *self.params)


@dataclass
class TimingConfig:
    """
    单标的日线择时配置。

    与选股策略分层一致：`timing.factor_list` 只负责算指标（走 step2 通用因子流水线），
    `timing` 只负责从指标列判买卖点。两层参数的分界是"改了要不要重算 step2"。
    """

    code: str
    code_type: str
    # 策略在组合中的最终资金权重，与选股策略的 cap_weight 字段对齐。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    cap_weight: float = 1.0
    name: str = "择时策略"  # 策略名，与选股策略的 name 字段对齐；框架会自动加序号前缀󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    rebalance_time: str = "open"  # 换仓时间，与选股策略的 rebalance_time 字段对齐󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    hold_period: str = INFINITE_HOLD_PERIOD  # 与选股策略的 hold_period 对齐，择时恒为 infinite󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # 信号库中的择时逻辑（timing 字段解析而来），含 name / factor_list / params󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    timing: TimingSignal = field(default_factory=TimingSignal)

    @classmethod
    def init(cls, index=0, **config) -> "TimingConfig":
        timing_raw = config.pop("timing", {})
        # 信号本身（名称、因子、参数、策略函数）全部由 TimingSignal 承载；󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 信号函数运行时通过传入的 TimingConfig 自行读取 factor_list󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        config["timing"] = TimingSignal.init(
            name=timing_raw.get("name", ""),
            factor_list=timing_raw.get("factor_list", []),
            params=timing_raw.get("params", ()),
        )

        name = str(config.get("name", "")).strip()
        config["name"] = name if name else config["timing"].name

        # 与选股策略共用同一套资金权重校验：必须是有限数值且不能为负数，默认 1󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        config["cap_weight"] = float(config.get("cap_weight", 1.0))

        rebalance_time = str(config.get("rebalance_time", "")).strip()
        config["rebalance_time"] = rebalance_time if rebalance_time else "open"
        parse_rebalance_time(config["rebalance_time"])

        config["code"] = cls._parse_code(config.get("code", ""))
        code_type = str(config.get("code_type", "")).strip().lower()
        if code_type not in {"stock", "etf", "index"}:
            raise ValueError("timing.code_type 只允许 stock、etf 或 index")
        if code_type == "index" and config["rebalance_time"] == "0935-0935":
            raise ValueError("指数日线数据不包含 09:35 价格，timing.rebalance_time 不支持 0935-0935")
        config["code_type"] = code_type

        # 择时依赖的指标必须显式声明在 timing.factor_list 中，用于 step2 计算和信号函数定位列名󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        if not config["timing"].factor_list:
            raise ValueError("timing.factor_list 必须是非空列表，用于声明择时逻辑依赖的指标")

        config.pop("hold_period", None)  # 择时的持仓周期恒为 infinite，不接受配置覆盖󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 保留与普通策略一致的展示名前缀。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        config["name"] = f"#{index}.{config['name']}"

        init_fields = {item.name for item in fields(cls) if item.init}
        if unknown_fields := config.keys() - init_fields:
            print(f"TimingConfig 已过滤未声明的配置项：{', '.join(sorted(unknown_fields))}")
            config = {key: value for key, value in config.items() if key in init_fields}

        return cls(**config)

    @staticmethod
    def _parse_code(raw_code) -> str:
        code = str(raw_code).strip()
        if len(code) == 8 and code[:2].lower() in {"sh", "sz", "bj"} and code[2:].isdigit():
            return code.lower()
        if len(code) == 9 and code[:6].isdigit() and code[6] == "." and code[7:].upper() in {"SH", "SZ", "BJ"}:
            return f"{code[7:].lower()}{code[:6]}"
        raise ValueError("timing.code 必须是 sh600000 或 600000.SH 这类有效代码")

    @property
    def all_factors(self) -> Tuple[FactorConfig, ...]:
        """返回择时依赖的去重因子，权重不参与去重。"""
        factor_dict = {}
        for factor in self.timing.factor_list:
            factor_dict.setdefault((factor.name, factor.param), factor)
        return tuple(factor_dict.values())

    @property
    def factor_columns(self) -> List[str]:
        return sorted({factor.col_name for factor in self.timing.factor_list})

    def get_timing_signal(self, stock_df: pd.DataFrame) -> pd.Series:
        """调用择时逻辑，返回与 stock_df 等长的 0/1 持仓状态。"""
        funcs = self.timing.funcs

        # 因子列必须在数据里存在，缺失说明需要重新运行 step2󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        missing = [factor.col_name for factor in self.timing.factor_list if factor.col_name not in stock_df.columns]
        if missing:
            raise KeyError(f"择时数据缺少因子列 {missing}，请重新运行 step2")

        # 信号函数自行从 self（TimingConfig）读取 factor_list 与 timing.params 解析因子列和参数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        signal = funcs["timing_signal"](self, stock_df)
        if not isinstance(signal, pd.Series):
            raise TypeError(f"{self.timing.name} 的 timing_signal() 必须返回 pandas.Series")
        if not signal.index.equals(stock_df.index):
            raise ValueError(f"{self.timing.name} 返回的信号索引必须与输入数据索引完全一致")

        invalid = signal.dropna()[~signal.dropna().isin([0, 1])]
        if not invalid.empty:
            raise ValueError(f"{self.timing.name} 的持仓状态只允许 0 和 1，实际包含：{invalid.unique()[:5].tolist()}")

        return signal.astype("float64").ffill().fillna(0.0).rename("持仓状态")

    def get_fullname(self) -> str:
        return f"{self.name}：{self.code}({self.code_type})，{self.timing.name}{self.timing.params}"

    def __repr__(self) -> str:
        return self.get_fullname()
