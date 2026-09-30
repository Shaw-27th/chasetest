"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import math
from copy import deepcopy
from datetime import datetime
from itertools import product
from numbers import Real
from pathlib import Path
from types import ModuleType
from typing import Optional, List

import numpy as np
import pandas as pd

from config import runtime_data_path
from core.model.strategy_config import HashableDict, StrategyConfig
from core.utils.factor_hub import FactorHub
from core.utils.path_kit import get_file_path, get_folder_path
from core.utils.strategy_hub import get_strategy_by_name
from core.market_essentials import get_trade_date, import_index_data
from core.model.rotation import RotationConfig
from core.model.timing_signal import EquityTiming, TimingConfig


class BacktestConfig:
    def __init__(self, **config_dict: dict):
        self.name: str = config_dict.get("backtest_name", "")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("backtest_name 必须是非空字符串")
        self.name = self.name.strip()

        self.start_date: Optional[str] = config_dict.get("start_date", None)  # 回测开始时间󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 回测结束时间，None 表示使用最新数据，也可指定日期，例如 '2022-11-01'󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.end_date: Optional[str] = config_dict.get("end_date", None)

        # 配置和模型都固定为两个策略，不提供通用策略列表入口。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.strategy_raw: dict = {}
        self.strategy2_raw: Optional[dict] = None
        self.strategy: Optional[StrategyConfig] = None
        self.strategy2: Optional[StrategyConfig] = None

        self.initial_cash: float = config_dict.get("initial_cash", 100_0000)  # 初始资金默认100万󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.c_rate: float = config_dict.get("c_rate", 1.2 / 10000)  # 手续费，默认为 0.00012，即万分之 1.2󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.t_rate: float = config_dict.get("t_rate", 1 / 1000)  # 印花税，默认为0.001󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

        self.data_center_path: Path = Path(str(config_dict["data_center_path"]))

        # 股票日线数据，全量数据下载链接：https://www.quantclass.cn/data/stock/stock-trading-data󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.stock_data_path: Path = Path(str(config_dict["stock_data_path"]))
        # ETF 日线数据，仅在单 ETF 日线择时时读取目标文件󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.etf_data_path: Path = Path(
            str(config_dict.get("etf_data_path", self.data_center_path / "stock-etf-trading-data"))
        )
        # 指数数据路径，全量数据下载链接：https://www.quantclass.cn/data/stock/stock-main-index-data󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.index_data_path: Path = Path(str(config_dict["index_data_path"]))
        # 财务数据，全量数据下载链接：https://www.quantclass.cn/data/stock/stock-fin-data-xbx󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.fin_data_path: Path = Path(str(config_dict.get("fin_data_path", "stock-fin-data-xbx")))

        self.has_fin_data: bool = self.fin_data_path.exists()  # 是否使用财务数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

        self.factor_params_dict: dict = {}  # 缓存策略和日线择时所需的全部因子参数󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.factor_col_name_list: List[str] = []
        self.hold_period_name_list: List[str] = []
        self.fin_cols: list = []  # 缓存财务因子列󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

        # 缓存被排除的板块󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.excluded_boards: list = config_dict.get("excluded_boards", [])

        # 资金曲线再择时配置，会在load_strategy中初始化󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.re_timing: Optional[EquityTiming] = None
        # 单标的日线择时配置（仅支持 strategy 槽位），会在 load_strategy 中初始化󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.timing_config: Optional[TimingConfig] = None
        # 多标的轮动配置（仅支持 strategy 槽位）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.rotation_config: Optional[RotationConfig] = None

        self.report: pd.DataFrame = pd.DataFrame()  # 回测报告󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self._runtime_cache: dict = {}

        # 遍历标记：遍历的INDEX，0表示非遍历场景，从1、2、3、4、...开始表示是第几个循环，当然也可以赋值为具体名称󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.iter_round: int | str = 0
        # 参数遍历任务名。遍历配置共用该名称对应的运行缓存，并按 Pro 版规则组织结果目录。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.factory_backtest_name: str = ""

    @property
    def is_timing_mode(self) -> bool:
        """是否运行单标的日线择时模式。"""
        return self.timing_config is not None

    @property
    def is_rotation_mode(self) -> bool:
        return self.rotation_config is not None

    @property
    def is_target_mode(self) -> bool:
        """是否只处理配置中明确列出的目标标的。"""
        return self.is_timing_mode or self.is_rotation_mode

    @property
    def target_code_type_map(self) -> dict[str, str]:
        if self.timing_config is not None:
            return {self.timing_config.code: self.timing_config.code_type}
        if self.rotation_config is not None:
            return {code: self.rotation_config.code_type for code in self.rotation_config.code_list}
        return {}

    def _reset_load_state(self, re_timing=None):
        """清空上一次加载产生的状态。"""
        self.strategy = None
        self.strategy2 = None
        self.timing_config = None
        self.rotation_config = None
        self.factor_params_dict = {}
        self.factor_col_name_list = []
        self.hold_period_name_list = []
        self.fin_cols = []
        self.re_timing = EquityTiming.init(**re_timing) if re_timing is not None else None

    def _load_rotation(self, rotation_raw, re_timing=None):
        """加载多标的轮动配置，并汇总仓位插件依赖的因子。"""
        self._reset_load_state(re_timing)
        weight = float(rotation_raw.get("cap_weight", 1))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("rotation.cap_weight 必须是非负有限数值")
        if weight < 1e-9:
            raise ValueError("没有读取到资金权重大于等于 1e-9 的轮动策略")

        rotation_dict = deepcopy(rotation_raw)
        rotation_dict["cap_weight"] = weight / max(weight, 1)
        self.rotation_config = RotationConfig.init(index=0, **rotation_dict)

        self.strategy_raw = {}
        self.strategy2_raw = None
        for factor_config in self.rotation_config.all_factors:
            self.factor_params_dict.setdefault(factor_config.name, set()).add(factor_config.param)
            self.factor_col_name_list.append(factor_config.col_name)
            self.fin_cols.extend(FactorHub.get_by_name(factor_config.name).fin_cols)
        self.factor_col_name_list = sorted(set(self.factor_col_name_list))
        self.hold_period_name_list = []
        self.fin_cols = sorted(set(self.fin_cols))
        if self.rotation_config.code_type in {"etf", "index"} and self.fin_cols:
            raise ValueError(f"ETF/指数轮动不支持财务因子，当前依赖：{self.fin_cols}")

    def _load_timing(self, timing_raw, re_timing=None):
        """
        加载 strategy 槽位中带 code 字段的单个日线择时配置。

        与 _load_strategy 平级：本方法只负责择时配置的加载，普通选股配置请调用
        _load_strategy；对外统一入口是 load_strategy，会按槽位内容自动分派。

        择时模式下不存在选股策略，因此会把 strategy_raw / strategy2_raw 置空，保持与
        self.strategy / self.strategy2（均为 None）的对应关系。
        """
        self._reset_load_state(re_timing)

        weight = float(timing_raw.get("cap_weight", 1))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("timing.cap_weight 必须是非负有限数值")
        if weight < 1e-9:
            raise ValueError("没有读取到资金权重大于等于 1e-9 的择时策略")

        timing_dict = deepcopy(timing_raw)
        timing_dict["cap_weight"] = weight / max(weight, 1)
        self.timing_config = TimingConfig.init(index=0, **timing_dict)

        # 择时模式的标的、信号和因子均由择时配置自身决定。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.strategy_raw = {}
        self.strategy2_raw = None
        for factor_config in self.timing_config.all_factors:
            self.factor_params_dict.setdefault(factor_config.name, set()).add(factor_config.param)
            self.factor_col_name_list.append(factor_config.col_name)
            self.fin_cols.extend(FactorHub.get_by_name(factor_config.name).fin_cols)

        self.factor_col_name_list = sorted(set(self.factor_col_name_list))
        # 择时按事件换仓，没有固定周期，因此不占用 period_offset 中的任何周期列󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.hold_period_name_list = []
        self.fin_cols = sorted(set(self.fin_cols))
        if self.timing_config.code_type in {"etf", "index"} and self.fin_cols:
            raise ValueError(f"ETF/指数日线择时不支持财务因子，当前运行依赖财务字段：{self.fin_cols}")

    def load_strategy(self, strategy, strategy2=None, re_timing=None):
        """按槽位内容自动分派加载配置，并汇总所需计算字段。

        对外统一入口：strategy 为轮动配置（含 rotation 字段）或择时配置（含 code 字段）时
        进入对应的单策略模式；strategy2 只用于普通选股双策略。
        """
        rotation_raw = strategy if isinstance(strategy, dict) and "rotation" in strategy else None
        timing_raw = strategy if isinstance(strategy, dict) and "code" in strategy else None
        strategy2_is_target = isinstance(strategy2, dict) and ("rotation" in strategy2 or "code" in strategy2)
        if strategy2_is_target:
            raise ValueError("轮动或择时配置只能写在 strategy，strategy2 仅支持普通选股策略")
        if rotation_raw is not None and timing_raw is not None:
            raise ValueError("轮动策略与单标的择时策略不能在同一次回测中混用")
        if rotation_raw is not None:
            if strategy2 is not None:
                raise ValueError("轮动模式仅支持单轮动，不能同时配置 strategy2")
            self._load_rotation(rotation_raw, re_timing)
            return
        if timing_raw is not None:
            if strategy2 is not None:
                raise ValueError("择时模式仅支持单择时，不能同时配置 strategy2")
            self._load_timing(timing_raw, re_timing)
            return
        self._load_strategy(strategy, strategy2, re_timing)

    def _load_strategy(self, strategy, strategy2=None, re_timing=None):
        """加载普通选股策略，并汇总所需计算字段。

        与 _load_timing 平级：本方法只负责普通选股配置（不含 code 字段的字典）的加载，
        日线择时配置请调用 _load_timing；对外统一入口是 load_strategy，会按槽位内容自动分派。
        """
        self._reset_load_state(re_timing)

        if not isinstance(strategy, dict):
            raise ValueError("strategy 必须存在且必须是字典")
        if "code" in strategy:
            raise ValueError("strategy 含 code 字段，是日线择时配置，请调用 load_strategy（自动分派）或 _load_timing")
        if "rotation" in strategy:
            raise ValueError("strategy 含 rotation 字段，是轮动配置，请调用 load_strategy（自动分派）或 _load_rotation")
        if strategy2 is not None and not isinstance(strategy2, dict):
            raise ValueError("strategy2 必须是字典")
        if strategy2 is not None and ("code" in strategy2 or "rotation" in strategy2):
            raise ValueError("strategy2 仅支持普通选股策略")

        self.strategy_raw = deepcopy(strategy)
        self.strategy2_raw = deepcopy(strategy2) if strategy2 is not None else None

        weight1 = self.strategy_raw.get("cap_weight", 1)
        weight2 = self.strategy2_raw.get("cap_weight", 1) if self.strategy2_raw is not None else 0
        for cap_weight in (weight1, weight2):
            if isinstance(cap_weight, bool) or not isinstance(cap_weight, Real) or not math.isfinite(cap_weight):
                raise ValueError("cap_weight 必须是有限数值")
            if cap_weight < 0:
                raise ValueError("cap_weight 不能为负数")

        weight1 = float(weight1)
        weight2 = float(weight2)
        all_cap_weight = max(weight1 + weight2, 1)
        name1 = self.strategy_raw.get("name")
        name2 = self.strategy2_raw.get("name") if self.strategy2_raw is not None else None
        if not isinstance(name1, str) or not name1.strip():
            raise ValueError("策略 name 必须是非空字符串")
        if self.strategy2_raw is not None and (not isinstance(name2, str) or not name2.strip()):
            raise ValueError("策略 name 必须是非空字符串")

        if weight1 >= 1e-9:
            stg_dict = deepcopy(self.strategy_raw)
            stg_dict["name"] = name1.strip()
            stg_dict["cap_weight"] = weight1 / all_cap_weight
            stg_dict["funcs"] = get_strategy_by_name(stg_dict["name"])
            self.strategy = StrategyConfig.init(index=0, **stg_dict)

        if self.strategy2_raw is not None and weight2 >= 1e-9:
            stg_dict = deepcopy(self.strategy2_raw)
            stg_dict["name"] = name2.strip()
            stg_dict["cap_weight"] = weight2 / all_cap_weight
            stg_dict["funcs"] = get_strategy_by_name(stg_dict["name"])
            self.strategy2 = StrategyConfig.init(index=1, **stg_dict)

        for loaded_strategy in (self.strategy, self.strategy2):
            if loaded_strategy is None:
                continue
            period_prefix = (
                loaded_strategy.hold_period[:-1]
                if loaded_strategy.hold_period.endswith("D")
                else loaded_strategy.hold_period
            )
            self.hold_period_name_list.append(f"{period_prefix}_0")
            self.factor_col_name_list.extend(loaded_strategy.factor_columns)
            for factor_config in loaded_strategy.all_factors:
                self.factor_params_dict.setdefault(factor_config.name, set()).add(factor_config.param)
                self.fin_cols.extend(FactorHub.get_by_name(factor_config.name).fin_cols)

        if self.strategy is None and self.strategy2 is None:
            raise ValueError("没有读取到资金权重大于等于 1e-9 的策略")

        self.factor_col_name_list = sorted(set(self.factor_col_name_list))
        self.hold_period_name_list = sorted(set(self.hold_period_name_list))
        self.fin_cols = sorted(set(self.fin_cols))

    def update_trading_date(self, tc_path):
        print("⚠️ 交易日历文件不存在，或者需要更新，从网络获取最新的交易日历数据。")
        index_data_all = import_index_data(self.index_data_path / "sh000001.csv")
        # 如果需要更新，从网络获取最新的交易日历数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        try:
            tc_df = get_trade_date(index_data_all)
            tc_df.to_csv(tc_path, index=False)
            print(f'🔄 交易日历更新为：{tc_df["交易日期"].min()}~{tc_df["交易日期"].max()}')
            return tc_df
        except Exception as e:
            print(e)
            print("需要更新交易日历，但无法联网")
            return None

    def read_index_with_trading_date(self, use_start_date=True):
        """
        加载交易日历和指数数据，补充周频、月频及日频周期标记。

        返回:
        DataFrame: 合并交易日历后的指数数据
        """
        # 获取今天的日期󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        today = datetime.today()
        start_date = self.start_date if use_start_date else None
        index_data = import_index_data(self.index_data_path / "sh000001.csv", [start_date, self.end_date])

        # 构建交易日历文件路径󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        tc_path = get_file_path(runtime_data_path, "交易日历.csv")

        if tc_path.exists():
            tc_df = pd.read_csv(tc_path)
            if pd.to_datetime(tc_df["交易日期"].max()) - today <= pd.to_timedelta("30 days"):
                new_tc_df = self.update_trading_date(tc_path)
                if new_tc_df is not None:
                    tc_df = new_tc_df
        else:
            tc_df = self.update_trading_date(tc_path)

        # 检查文件是否存在，或者是否需要更新󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        if tc_df is None:
            print("本地不存在交易日历，需要联网更新后继续，程序退出")
            exit()

        print(f'🌀 本地交易日历数据为：{tc_df["交易日期"].min()}~{tc_df["交易日期"].max()}')

        # 将交易日期列转换为datetime类型󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        tc_df["交易日期"] = pd.to_datetime(tc_df["交易日期"])

        # 计算下个交易日󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        tc_df["次交易日"] = tc_df["交易日期"].shift(-1)

        # 标记周频起始日󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        con1 = tc_df["交易日期"].diff().dt.days != 1
        tc_df.loc[con1, "周频起始日"] = tc_df["交易日期"]

        # 处理只有一个交易日的周期󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        con2 = tc_df["交易日期"].diff(-1).dt.days != -1
        tc_df.loc[con1 & con2, "周频起始日"] = np.nan
        tc_df["周频起始日"] = tc_df["周频起始日"].ffill()
        tc_df["周频终止日"] = tc_df["周频起始日"] != tc_df["周频起始日"].shift(-1)

        # 标记月频起始日󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        con3 = tc_df["交易日期"].dt.month != tc_df["交易日期"].shift().dt.month
        tc_df.loc[con3, "月频起始日"] = tc_df["交易日期"]
        tc_df["月频起始日"] = tc_df["月频起始日"].ffill()
        tc_df["月频终止日"] = tc_df["月频起始日"] != tc_df["月频起始日"].shift(-1)

        # ==标记3D、5D、10D的开始和截止日期󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 日频系列需要指定一个基础的交易日期，我们指定2007年第一个交易日期作为指定日期（2007-01-04）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        base_inx = tc_df[tc_df["交易日期"] == pd.to_datetime("2007-01-04")].index.min()
        if pd.isnull(base_inx):
            print(f'🚨 删除: {get_file_path(runtime_data_path, "交易日历.csv")}')
            raise Exception(
                "交易日历至少需要从2007年1月4日开始，请删除runtime_data_path目录下的`交易日历.csv`，"
                "并确保sh000001指数至少从2007年1月4日开始!"
                "\n指数文件路径：%s" % self.index_data_path
            )
        # 计算不同周期的起始日期󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        for n in [3, 5, 10]:
            con = (tc_df.index - base_inx) % n == 0
            tc_df.loc[con, f"{n}D起始日"] = tc_df["交易日期"]
            tc_df[f"{n}D起始日"] = tc_df[f"{n}D起始日"].ffill()
            tc_df[f"{n}D终止日"] = tc_df[f"{n}D起始日"] != tc_df[f"{n}D起始日"].shift(-1)

        # 将交易日历数据与指数数据合并󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        index_data = pd.merge(left=index_data, right=tc_df, on="交易日期", how="left")

        # 额外生成实盘需要的周期数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        period_offset = tc_df[["交易日期"]]
        for period, tag in {"周频": "W_0", "月频": "M_0", "3D": "3_0", "5D": "5_0", "10D": "10_0"}.items():
            period_offset[tag] = 0
            period_offset.loc[period_offset["交易日期"] == tc_df[f"{period}起始日"], tag] = 1
            period_offset[tag] = period_offset[tag].cumsum()

        period_offset_path = get_file_path(runtime_data_path, "period_offset.csv")
        period_offset.columns = pd.MultiIndex.from_tuples(
            zip(
                ["数据由整理，对数据字段有疑问的，可以直接微信私信邢不行，微信号：xbx297"]
                + [""] * (period_offset.shape[1] - 1),
                period_offset.columns,
            )
        )
        period_offset.to_csv(period_offset_path, encoding="gbk", index=False)
        return index_data

    def get_result_folder(self) -> Path:
        if self.iter_round == 0:
            return get_folder_path(runtime_data_path, "回测结果", self.name)

        factory_name = self.factory_backtest_name or self.name
        config_name = f"S{self.iter_round}-{factory_name}" if isinstance(self.iter_round, int) else self.iter_round
        if self.name.startswith(f"S{self.iter_round}"):
            config_name = self.name
        return get_folder_path(runtime_data_path, "遍历结果", factory_name, config_name, path_type=True)

    def get_runtime_folder(self) -> Path:
        cache_name = self.factory_backtest_name if self.iter_round != 0 and self.factory_backtest_name else self.name
        return get_folder_path(runtime_data_path, "运行缓存", cache_name)

    def cache_set(self, key: str, value):
        self._runtime_cache[key] = value

    def cache_get(self, key: str, default=None):
        return self._runtime_cache.get(key, default)

    def cache_del(self, key: str):
        self._runtime_cache.pop(key, None)

    def cache_clear(self):
        self._runtime_cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_runtime_cache"] = {}
        return state

    def read_trading_dates(self, start_date, end_date):
        calendar = pd.read_csv(
            get_file_path(runtime_data_path, "交易日历.csv"), encoding="utf-8", parse_dates=["交易日期"]
        )
        trading_dates = calendar["交易日期"]
        return trading_dates[(trading_dates >= start_date) & (trading_dates <= pd.to_datetime(end_date))]

    def load_period_offset(self) -> pd.DataFrame:
        """读取双策略模拟所需的周期编号，只加载当前有效策略使用的周期。"""
        return pd.read_csv(
            get_file_path(runtime_data_path, "period_offset.csv"),
            encoding="gbk",
            parse_dates=["交易日期"],
            skiprows=1,
            usecols=["交易日期", *self.hold_period_name_list],
        )

    def get_fullname(self):
        if self.rotation_config is not None:
            rotation_desc = f"{self.rotation_config.cap_weight:.2%} {self.rotation_config.get_fullname()}"
            fullname = f"{self.name}：轮动：{rotation_desc}，初始资金￥{self.initial_cash:,.2f}"
        elif self.timing_config is not None:
            timing_desc = f"{self.timing_config.cap_weight:.2%} {self.timing_config.get_fullname()}"
            fullname = f"{self.name}：个股择时：{timing_desc}，初始资金￥{self.initial_cash:,.2f}"
        else:
            strategy_desc = "；".join(
                f"{strategy.cap_weight:.2%} {strategy.get_fullname()}"
                for strategy in (self.strategy, self.strategy2)
                if strategy is not None
            )
            fullname = f"{self.name}：{strategy_desc}，初始资金￥{self.initial_cash:,.2f}"
        if self.re_timing is not None:
            fullname += f"，再择时：{self.re_timing.name, self.re_timing.params}"
        return fullname

    def set_report(self, report: pd.DataFrame):
        report["param"] = self.get_fullname()
        self.report = report

    def get_strategy_config_sheet(self, with_factors=True) -> dict:
        if self.rotation_config is not None:
            ret = {"策略": self.name, "策略详情": self.get_fullname()}
            rotation = self.rotation_config
            ret["轮动标的"] = "，".join(rotation.code_list)
            ret["标的类型"] = rotation.code_type
            ret["换仓时间"] = rotation.rebalance_time
            ret["资金权重"] = f"{rotation.cap_weight:.2%}"
            ret["仓位策略"] = rotation.rotation.name
            ret["最大持有数"] = rotation.rotation.max_select_num
            ret["仓位参数"] = str(rotation.rotation.params)
            if with_factors:
                for factor_config in rotation.all_factors:
                    ret[f"#因子-{factor_config.name}"] = str(factor_config.param)
            return ret
        if self.timing_config is not None:
            ret = {"策略": self.name, "策略详情": self.get_fullname()}
            timing_config = self.timing_config
            ret["择时标的"] = timing_config.code
            ret["标的类型"] = timing_config.code_type
            ret["换仓时间"] = timing_config.rebalance_time
            ret["持仓周期"] = timing_config.hold_period
            ret["资金权重"] = f"{timing_config.cap_weight:.2%}"
            ret["择时信号"] = timing_config.timing.name
            ret["择时参数"] = str(timing_config.timing.params)
            if with_factors:
                for factor_config in timing_config.all_factors:
                    factor_key = f"#因子-{factor_config.name}"
                    factor_param = str(factor_config.param)
                    if factor_key in ret:
                        ret[factor_key] = f"{ret[factor_key]}，{factor_param}"
                    else:
                        ret[factor_key] = factor_param
            return ret

        strategies = tuple(strategy for strategy in (self.strategy, self.strategy2) if strategy is not None)
        factor_dict = {
            "持仓周期": "，".join(strategy.hold_period for strategy in strategies),
            "选股数量": "，".join(str(strategy.select_num) for strategy in strategies),
            "换仓时间": "，".join(strategy.rebalance_time for strategy in strategies),
            "资金权重": "，".join(f"{strategy.cap_weight:.2%}" for strategy in strategies),
        }
        ret = {"策略": self.name, "策略详情": self.get_fullname()}
        if with_factors:
            for strategy in strategies:
                for factor_config in strategy.all_factors:
                    _name = f"#因子-{factor_config.name}"
                    _val = str(factor_config.param)
                    if _name in factor_dict:
                        factor_dict[_name] = f"{factor_dict[_name]}，{_val}"
                    else:
                        factor_dict[_name] = _val
            ret.update(**factor_dict)

        return ret

    @classmethod
    def init_from_config(cls, load_strategy=True):
        import config

        # 提取自定义变量󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        config_dict = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("__") and not isinstance(value, ModuleType)
        }
        conf = cls(**config_dict)

        config_vars = vars(config)
        strategy_raw = deepcopy(config_vars.get("strategy"))
        strategy2_raw = deepcopy(config_vars.get("strategy2"))
        # 轮动/择时/普通选股的判断与分派统一交给 load_strategy。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 轮动和择时只支持 strategy，strategy2 仅用于普通选股双策略。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        conf.strategy_raw = strategy_raw
        conf.strategy2_raw = strategy2_raw
        if load_strategy:
            conf.load_strategy(strategy_raw, strategy2_raw, config_vars.get("re_timing"))
        return conf


class BacktestConfigFactory:
    """
    遍历参数的时候，动态生成配置
    """

    def __init__(self, backtest_name: str):
        # ====================================================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # ** 参数遍历配置 **󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 可以指定因子遍历的参数范围󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # ====================================================================================================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 存储生成好的config list和strategy list󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.config_list: List[BacktestConfig] = []
        self.backtest_name = backtest_name

        # 存储所有因子的配置，dummy的哦󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        self.full_conf: Optional[BacktestConfig] = None

    @property
    def result_folder(self) -> Path:
        return get_folder_path(runtime_data_path, "遍历结果", self.backtest_name, path_type=True)

    def generate_all_factor_config(self):
        """
        产生一个conf，拥有所有策略的因子，用于因子加速并行计算

        普通选股模式下因子位于 strategy_raw["factor_list"] / ["filter_list"]；
        日线择时/轮动模式下因子位于各自专用配置中，需要重建合并全部因子参数的模板配置。
        """
        backtest_config = BacktestConfig.init_from_config(load_strategy=False)
        backtest_config.name = self.backtest_name

        first_conf = self.config_list[0]

        # —— 轮动模式：合并全部参数组合的标的池和因子，只准备、计算一次 ——󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        if first_conf.rotation_config is not None:
            merged_codes = []
            merged_factor_tuples = []
            seen_codes = set()
            seen_factors = set()
            for conf in self.config_list:
                rotation_config = conf.rotation_config
                for code in rotation_config.code_list:
                    if code not in seen_codes:
                        seen_codes.add(code)
                        merged_codes.append(code)
                for factor in rotation_config.all_factors:
                    key = (factor.name, factor.param)
                    if key not in seen_factors:
                        seen_factors.add(key)
                        merged_factor_tuples.append(factor.to_tuple())

            rotation0 = first_conf.rotation_config
            strategy = {
                "code_list": merged_codes,
                "code_type": rotation0.code_type,
                "cap_weight": rotation0.cap_weight,
                "name": str(rotation0.name).split(".", 1)[-1],
                "rebalance_time": rotation0.rebalance_time,
                "rotation": {
                    "name": rotation0.rotation.name,
                    "factor_list": merged_factor_tuples,
                    "max_select_num": rotation0.rotation.max_select_num,
                    "params": rotation0.rotation.params,
                },
            }
            backtest_config._load_rotation(strategy)
            return backtest_config

        # —— 日线择时模式：合并所有参数组合的择时因子，只计算一次 ——󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        if first_conf.timing_config is not None:
            merged_factor_tuples = []
            seen = set()
            for conf in self.config_list:
                for factor in conf.timing_config.all_factors:
                    key = (factor.name, factor.param)
                    if key not in seen:
                        seen.add(key)
                        merged_factor_tuples.append(factor.to_tuple())

            # 以第一个择时配置为模板重建原始配置字典（name 去掉展示用序号前缀，避免二次加载叠加前缀）󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            timing0 = self.config_list[0].timing_config
            name = str(timing0.name).split(".", 1)[-1]
            raw_params = timing0.timing.params
            if isinstance(raw_params, HashableDict):
                raw_params = dict(raw_params.data)
            strategy = {
                "code": timing0.code,
                "code_type": timing0.code_type,
                "cap_weight": timing0.cap_weight,
                "name": name,
                "rebalance_time": timing0.rebalance_time,
                "timing": {"name": timing0.timing.name, "factor_list": merged_factor_tuples, "params": raw_params},
            }
            backtest_config._load_timing(strategy)
            return backtest_config

        # —— 普通选股模式：原有逻辑 ——󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        factor_list = [[], []]
        filter_list = [[], []]
        for conf in self.config_list:
            for index, strategy_raw in enumerate((conf.strategy_raw, conf.strategy2_raw)):
                if strategy_raw is None:
                    continue
                factor_list[index] += strategy_raw["factor_list"]
                filter_list[index] += strategy_raw["filter_list"]

        # 参数遍历的策略以第一组为模板，因子列表则合并所有组合，从而只计算一次因子。󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        strategy = {**first_conf.strategy_raw, "factor_list": factor_list[0], "filter_list": filter_list[0]}
        strategy2 = None
        if first_conf.strategy2_raw is not None:
            strategy2 = {**first_conf.strategy2_raw, "factor_list": factor_list[1], "filter_list": filter_list[1]}
        backtest_config._load_strategy(strategy, strategy2)
        return backtest_config

    def get_name_params_sheet(self) -> pd.DataFrame:
        rows = []
        for config in self.config_list:
            rows.append(config.get_strategy_config_sheet())

        sheet = pd.DataFrame(rows)
        sheet.to_excel(self.config_list[-1].get_result_folder().parent / "策略回测参数总表.xlsx", index=False)
        return sheet

    def generate_by_strategies(self, strategies, equity_signals=(None,)) -> List[BacktestConfig]:
        config_list = []
        iter_round = 0

        for strategy_group, equity_signal in product(strategies, equity_signals):
            iter_round += 1
            if isinstance(strategy_group, tuple):
                strategy, strategy2 = strategy_group
            else:
                strategy, strategy2 = strategy_group, None
            backtest_config = BacktestConfig.init_from_config(load_strategy=False)
            backtest_config.load_strategy(strategy, strategy2, re_timing=equity_signal)
            backtest_config.iter_round = iter_round
            backtest_config.factory_backtest_name = self.backtest_name
            backtest_config.name = f"S{iter_round}-{self.backtest_name}"

            config_list.append(backtest_config)

        self.config_list = config_list

        self.full_conf = self.generate_all_factor_config()

        return config_list


def load_config() -> BacktestConfig:
    return BacktestConfig.init_from_config()


def create_factory(strategies, backtest_name=None, re_timing_strategies=(None,)):
    if backtest_name is None:
        from config import backtest_name
    if not re_timing_strategies:
        re_timing_strategies = (None,)
    factory = BacktestConfigFactory(backtest_name)
    factory.generate_by_strategies(strategies, re_timing_strategies)

    return factory
