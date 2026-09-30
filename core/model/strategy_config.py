"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import re
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import List, Dict, Callable

import numpy as np
import pandas as pd

import config
from config import days_listed
from core.model.type_def import parse_rebalance_time
from core.utils.strategy_hub import get_pos_strategy_by_name


def filter_series_by_range(series, range_str):
    # 提取运算符和数值󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    operator = range_str[:2] if range_str[:2] in [">=", "<=", "==", "!="] else range_str[0]
    value = float(range_str[len(operator) :])

    match operator:
        case ">=":
            return series >= value
        case "<=":
            return series <= value
        case "==":
            return series == value
        case "!=":
            return series != value
        case ">":
            return series > value
        case "<":
            return series < value
        case _:
            raise ValueError(f"Unsupported operator: {operator}")


def get_col_name(factor_name, factor_param):
    return f"{factor_name}_{str(factor_param)}"


# 自定义一个类来保持dict的使用方法，并保证其可哈希，且保证顺序󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
class HashableDict:
    def __init__(self, data: dict):
        # 将字典按键排序并转为tuple，保证顺序并可哈希󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.data = tuple(sorted(data.items()))

    def __repr__(self):
        # 使其返回一个类似字典的表示方式󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        if isinstance(self.data, tuple):
            return "{" + ", ".join(f"{k}: {v}" for k, v in self.data) + "}"
        return repr(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    # 支持通过 [] 方式访问󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    def __getitem__(self, key):
        if isinstance(self.data, tuple):
            # 将tuple转换回一个dict来支持按键访问󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            dict_data = dict(self.data)
            return dict_data[key]
        else:
            raise TypeError(f"Cannot subscript a {type(self.data)} object")


def parse_param(param) -> tuple | HashableDict | str | int | float | bool | None:
    # param的类型需要转换为hashable的状态󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    if isinstance(param, list):
        param = tuple(param)
    elif isinstance(param, dict):
        param = HashableDict(param)
    elif isinstance(param, (str, int, float, tuple, bool)) or param is None:
        pass
    else:
        raise ValueError(f"不支持的参数类型：{type(param)}")
    return param


@dataclass(frozen=True)
class FactorConfig:
    name: str = "Bias"  # 选股因子名称󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    is_sort_asc: bool = True  # 是否正排序󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    param: tuple | HashableDict | str | int | float | bool | None = 3  # 选股因子参数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    weight: float = 1  # 选股因子权重󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    @classmethod
    def parse_list(cls, config_list: List[tuple]):
        all_long_factor_weight = sum([factor[3] for factor in config_list])
        factor_list = []
        for factor_name, is_sort_asc, param, weight in config_list:
            new_weight = weight / all_long_factor_weight
            # param的类型需要转换为hashable的状态󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            param = parse_param(param)
            factor_list.append(cls(name=factor_name, is_sort_asc=is_sort_asc, param=param, weight=new_weight))
        return factor_list

    @cached_property
    def col_name(self):
        return get_col_name(self.name, self.param)

    def __repr__(self):
        return f'{self.col_name}{"↑" if self.is_sort_asc else "↓"}#权重:{self.weight}'

    def to_tuple(self):
        return self.name, self.is_sort_asc, self.param, self.weight


@dataclass(frozen=True)
class FilterMethod:
    how: str = ""  # 过滤方式󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    range: str = ""  # 过滤值󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    def __repr__(self):
        match self.how:
            case "rank":
                name = "排名"
            case "pct":
                name = "排名百分比"
            case "val":
                name = "数值"
            case _:
                raise ValueError(f"不支持的过滤方式：`{self.how}`")

        return f"{name}:{self.range}"

    def to_val(self):
        return f"{self.how}:{self.range}"


@dataclass(frozen=True)
class FilterFactorConfig:
    name: str = "Bias"  # 选股因子名称󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    param: tuple | HashableDict | str | int | float | bool | None = 3  # 选股因子参数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    method: FilterMethod = None  # 过滤方式󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    is_sort_asc: bool = True  # 是否正排序󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    def __repr__(self):
        _repr = self.col_name
        if self.method:
            _repr += f'{"↑" if self.is_sort_asc else "↓"}#{self.method}'
        return _repr

    @cached_property
    def col_name(self):
        return get_col_name(self.name, self.param)

    @classmethod
    def init(cls, filter_factor: tuple):
        # 仔细看，结合class的默认值，这个和默认策略中使用的过滤是一模一样的󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        config = dict(name=filter_factor[0], param=parse_param(filter_factor[1]))
        if len(filter_factor) > 2:
            # 可以自定义过滤方式󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            _how, _range = re.sub(r"\s+", "", filter_factor[2]).split(":")
            config["method"] = FilterMethod(how=_how, range=_range)
        if len(filter_factor) > 3:
            # 可以自定义排序󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            config["is_sort_asc"] = filter_factor[3]
        return cls(**config)

    def to_tuple(self, full_mode=False):
        if full_mode:
            return self.name, self.param, self.method.to_val(), self.is_sort_asc
        else:
            return self.name, self.param


def calc_factor_common(df, factor_list: List[FactorConfig]):
    factor_val = np.zeros(df.shape[0])
    for factor_config in factor_list:
        # 计算单个因子的排名󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        _rank = df.groupby("交易日期")[factor_config.col_name].rank(ascending=factor_config.is_sort_asc, method="min")
        # 将因子按照权重累加󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        factor_val += _rank * factor_config.weight
    return factor_val


def filter_common(df, filter_list):
    condition = pd.Series(True, index=df.index)

    for filter_config in filter_list:
        col_name = f"{filter_config.name}_{str(filter_config.param)}"
        match filter_config.method.how:
            case "rank":
                rank = df.groupby("交易日期")[col_name].rank(ascending=filter_config.is_sort_asc, pct=False)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case "pct":
                rank = df.groupby("交易日期")[col_name].rank(ascending=filter_config.is_sort_asc, pct=True)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case "val":
                condition = condition & filter_series_by_range(df[col_name], filter_config.method.range)
            case _:
                raise ValueError(f"不支持的过滤方式：{filter_config.method.how}")

    return condition


@dataclass
class StrategyConfig:
    # 策略展示名；框架会自动加序号前缀（#0./#1.），保证多策略展示名唯一󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    name: str = "Strategy"

    # 策略在组合中的最终资金权重，由 BacktestConfig 统一校验和归一化。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    cap_weight: float = 1.0

    # 持仓周期。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    hold_period: str = "W"

    # 原始数据的周期。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    candle_period: str = "D"

    # 选股数量。1 表示一个股票; 0.1 表示做多10%的股票󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    select_num: int or float = 0.1

    # 换仓时间。可选值见 type_def.SUPPORTED_REBALANCE_TIMES，目前只开放三种：󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    # open 次日开盘调仓，close-open 当日收盘卖出、次日开盘买入，0935-0935 次日 9:35 调仓󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    rebalance_time: str = "open"

    # 因子列名。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    factor_name: str = "复合因子"  # 默认使用 factor_list 中的因子组合；自定义时需配合实现 calc_select_factor 函数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    factor_list: List[FactorConfig] = field(
        default_factory=list
    )  # 因子名（与“因子库”中的文件名一致），排序方式，参数，权重󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    # 过滤因子名（与“因子库”中的文件名一致），参数，过滤规则，排序方式󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    filter_list: List[FilterFactorConfig] = field(default_factory=list)

    # 策略函数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    funcs: Dict[str, Callable] = field(default_factory=dict)

    @cached_property
    def period_type(self) -> str:
        return self.hold_period[-1]

    @cached_property
    def hold_period_name(self) -> str:
        match self.period_type:
            case "W":
                return "周频"
            case "M":
                return "月频"
            case _:
                return self.hold_period

    @cached_property
    def factor_columns(self) -> List[str]:
        factor_columns = set()  # 去重󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

        # 针对当前策略的因子信息，整理之后的列名信息，并且缓存到全局󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        for factor_config in self.factor_list:
            # 策略因子最终在df中的列名󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            factor_columns.add(factor_config.col_name)  # 添加到当前策略缓存信息中󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

        # 针对当前策略的过滤因子信息，整理之后的列名信息，并且缓存到全局󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        for filter_factor in self.filter_list:
            # 策略过滤因子最终在df中的列名󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            factor_columns.add(filter_factor.col_name)  # 添加到当前策略缓存信息中󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

        return list(factor_columns)

    @cached_property
    def all_factors(self) -> set:
        all_factors = set()
        for factor_config in self.factor_list:
            all_factors.add(factor_config)
        for filter_factor in self.filter_list:
            all_factors.add(filter_factor)
        return all_factors

    @classmethod
    def init(cls, index=0, **config):
        config["factor_list"] = FactorConfig.parse_list(config.get("factor_list", []))
        config["filter_list"] = [
            FilterFactorConfig.init(filter_config) for filter_config in config.get("filter_list", [])
        ]

        init_fields = {item.name for item in fields(cls) if item.init}
        if unknown_fields := config.keys() - init_fields:
            print(f"StrategyConfig 已过滤未声明的配置项：{', '.join(sorted(unknown_fields))}")
            config = {key: value for key, value in config.items() if key in init_fields}

        stg_conf = cls(**config)
        # 展示名自动加序号前缀（#0./#1.），保证多策略展示名唯一，模块加载仍使用配置里的原始 name󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        stg_conf.name = f"#{index}.{stg_conf.name}"
        # 换仓时间在配置加载阶段就解析一次，把拼写错误挡在 step4 之前󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        try:
            parse_rebalance_time(stg_conf.rebalance_time)
        except ValueError as e:
            raise ValueError(f"策略 {stg_conf.name} 的换仓时间配置错误：{e}") from e
        return stg_conf

    def __repr__(self):
        return f"{self.name}-{self.hold_period}-{self.select_num}-{self.factor_list}-{self.filter_list}"

    def get_fullname(self):
        return (
            f"{self.name}-周期{self.hold_period}-数量{self.select_num}-换仓{self.rebalance_time}"
            f"-因子: {self.factor_list}-过滤：{self.filter_list}"
        )

    def max_int_param(self) -> int:
        max_int = 0
        for factor_config in self.all_factors:
            if isinstance(factor_config.param, int):
                max_int = max(max_int, factor_config.param)
        return max_int

    @staticmethod
    def filter_after_condition(period_df):
        cond6 = period_df["下日_是否交易"] == 1
        cond7 = period_df["下日_一字涨停"] != 1 if config.stay_real else period_df["下日_开盘涨停"] != 1
        cond8 = period_df["下日_是否ST"] != 1
        cond9 = period_df["下日_是否退市"] != 1
        return cond6 & cond7 & cond8 & cond9

    def filter_before_select(self, period_df):
        if "filter_stock" in self.funcs:
            return self.funcs["filter_stock"](period_df)

        # 通用的filter筛选󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # =删除不能交易的周期数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 删除月末为st状态的周期数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        cond1 = ~period_df["股票名称"].str.contains("ST", regex=False)
        # 删除月末为s状态的周期数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        cond2 = ~period_df["股票名称"].str.contains("S", regex=False)
        # 删除月末有退市风险的周期数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        cond3 = ~period_df["股票名称"].str.contains("*", regex=False)
        cond4 = ~period_df["股票名称"].str.contains("退", regex=False)
        # 删除交易天数过少的周期数  （已注释，未来函数）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # cond5 = period_df["交易天数"] / period_df["市场交易天数"] >= 0.8󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

        cond10 = period_df["上市至今交易天数"] > days_listed

        common_filter = cond1 & cond2 & cond3 & cond4 & cond10
        if not config.stay_real:
            common_filter &= self.filter_after_condition(period_df)
        period_df = period_df[common_filter]

        filter_condition = filter_common(period_df, self.filter_list)

        return period_df[filter_condition]

    def filter_after_select(self, period_df):
        if config.stay_real:
            period_df = period_df[self.filter_after_condition(period_df)]
        return period_df

    def calc_select_factor(self, period_df):
        if "calc_select_factor" in self.funcs:
            return self.funcs["calc_select_factor"](period_df)
        new_cols = {self.factor_name: self.calc_select_factor_default(period_df)}
        return pd.DataFrame(new_cols, index=period_df.index)

    def calc_select_factor_default(self, period_df):
        return calc_factor_common(period_df, self.factor_list)


@dataclass
class PosStrategyConfig:
    name: str
    factor_list: tuple[FactorConfig, ...]
    max_select_num: int = 1
    params: dict = field(default_factory=dict)
    core_funcs: dict = field(default_factory=dict)

    @classmethod
    def init(cls, **config) -> "PosStrategyConfig":
        name = str(config.get("name", "")).strip()
        if not name:
            raise ValueError("rotation.name 必须是非空字符串")
        params = config.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("rotation.params 必须是字典")
        factor_list = tuple(FactorConfig.parse_list(config.get("factor_list", [])))
        if not factor_list:
            raise ValueError("rotation.factor_list 必须是非空列表")
        max_select_num = config.get("max_select_num", 1)
        if isinstance(max_select_num, bool) or not isinstance(max_select_num, int) or max_select_num < 1:
            raise ValueError("rotation.max_select_num 必须是正整数")
        return cls(
            name=name,
            factor_list=factor_list,
            max_select_num=max_select_num,
            params=params.copy(),
            core_funcs=get_pos_strategy_by_name(name),
        )

    @property
    def factor_columns(self) -> list[str]:
        return sorted({factor.col_name for factor in self.factor_list})

    def calc_ratios(self, assets: List[dict]) -> pd.DataFrame:
        return self.core_funcs["calc_ratio"](assets, stg_conf=self)

    def __repr__(self):
        return f"{self.name}-最多选择{self.max_select_num}个-因子{self.factor_list}-参数{self.params}"
