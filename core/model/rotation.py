"""多标的日线轮动配置模型。"""

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from core.model.strategy_config import FactorConfig, PosStrategyConfig
from core.model.timing_signal import INFINITE_HOLD_PERIOD
from core.model.type_def import parse_rebalance_time

@dataclass
class RotationConfig:
    code_list: tuple[str, ...]
    code_type: str
    rotation: PosStrategyConfig
    cap_weight: float = 1.0
    name: str = "轮动策略"
    rebalance_time: str = "open"
    hold_period: str = INFINITE_HOLD_PERIOD

    @classmethod
    def init(cls, index=0, **config) -> "RotationConfig":
        rotation = PosStrategyConfig.init(**config.get("rotation", {}))
        raw_codes = config.get("code_list", [])
        if not isinstance(raw_codes, (list, tuple)) or len(raw_codes) < 2:
            raise ValueError("rotation.code_list 至少需要两个标的")
        code_list = tuple(cls._parse_code(code) for code in raw_codes)
        if len(set(code_list)) != len(code_list):
            raise ValueError("rotation.code_list 不允许包含重复标的")
        if rotation.max_select_num > len(code_list):
            raise ValueError("rotation.max_select_num 不能超过 code_list 的标的数量")

        code_type = str(config.get("code_type", "")).strip().lower()
        if code_type not in {"stock", "etf", "index"}:
            raise ValueError("rotation.code_type 只允许 stock、etf 或 index")

        cap_weight = float(config.get("cap_weight", 1.0))
        if not isfinite(cap_weight) or cap_weight < 0:
            raise ValueError("rotation.cap_weight 必须是非负有限数值")

        rebalance_time = str(config.get("rebalance_time", "open")).strip() or "open"
        parse_rebalance_time(rebalance_time)
        if code_type == "index" and rebalance_time == "0935-0935":
            raise ValueError("指数日线数据不包含 09:35 价格，rotation.rebalance_time 不支持 0935-0935")
        name = str(config.get("name", "")).strip() or rotation.name
        return cls(
            code_list=code_list,
            code_type=code_type,
            rotation=rotation,
            cap_weight=cap_weight,
            name=f"#{index}.{name}",
            rebalance_time=rebalance_time,
        )

    @staticmethod
    def _parse_code(raw_code) -> str:
        code = str(raw_code).strip()
        if len(code) == 8 and code[:2].lower() in {"sh", "sz", "bj"} and code[2:].isdigit():
            return code.lower()
        if len(code) == 9 and code[:6].isdigit() and code[6] == "." and code[7:].upper() in {"SH", "SZ", "BJ"}:
            return f"{code[7:].lower()}{code[:6]}"
        raise ValueError("rotation.code_list 中的代码必须是 sh000300 或 000300.SH 这类格式")

    @property
    def all_factors(self) -> tuple[FactorConfig, ...]:
        factor_dict = {}
        for factor in self.rotation.factor_list:
            factor_dict.setdefault((factor.name, factor.param), factor)
        return tuple(factor_dict.values())

    @property
    def factor_columns(self) -> list[str]:
        return sorted({factor.col_name for factor in self.rotation.factor_list})

    def get_ratio(self, assets: list[dict]) -> pd.DataFrame:
        ratio = self.rotation.calc_ratios(assets)
        if not isinstance(ratio, pd.DataFrame):
            raise TypeError(f"{self.rotation.name}.calc_ratio() 必须返回 pandas.DataFrame")
        if list(ratio.columns) != list(self.code_list):
            raise ValueError("轮动仓位结果的列必须与 code_list 完全一致且顺序相同")
        if ratio.index.has_duplicates or not ratio.index.is_monotonic_increasing:
            raise ValueError("轮动仓位结果的交易日期索引必须唯一且升序")
        if ratio.isna().any().any() or ((ratio < 0) | (ratio > 1)).any().any():
            raise ValueError("轮动仓位只允许 0 到 1 之间的有限数值")
        if (ratio.sum(axis=1) > 1 + 1e-9).any():
            raise ValueError("轮动仓位每日合计不能超过 1")
        return ratio.astype("float64")

    def get_fullname(self) -> str:
        return (
            f"{self.name}：{self.rotation.name}{self.rotation.params}，"
            f"最多持有{self.rotation.max_select_num}个，标的数{len(self.code_list)}"
        )
#󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
