"""多周期动量轮动：按横截面动量排名生成目标仓位。"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import pandas as pd


if TYPE_CHECKING:
    from core.model.strategy_config import PosStrategyConfig


def calc_ratio(assets: List[dict], stg_conf: "PosStrategyConfig") -> pd.DataFrame:
    """返回以交易日期为索引、轮动标的为列的目标资金占比。"""
    factor = stg_conf.factor_list[0]
    params = stg_conf.params
    max_select_num = stg_conf.max_select_num

    factor_panel = pd.concat(
        {
            asset["name"]: asset["df"].set_index("交易日期")[factor.col_name]
            for asset in assets
        },
        axis=1,
        join="outer",
    )
    factor_panel.sort_index(inplace=True)
    ratios = pd.DataFrame(0.0, index=factor_panel.index, columns=factor_panel.columns)

    for trade_date, values in factor_panel.iterrows():
        valid = values.dropna()
        if valid.empty:
            continue

        # 动量因子通常是“越大越好”，is_sort_asc = False 表示从高到低排序󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        ascending = factor.is_sort_asc
        best_value = valid.min() if ascending else valid.max()
        if params.get("empty_when_all_negative", False) and best_value < 0:
            continue

        select_num = min(max_select_num, len(valid))
        selected = valid.sort_values(ascending=ascending, kind="stable").index[:select_num]
        ratios.loc[trade_date, selected] = 1.0 / len(selected)
    print(ratios)
    return ratios