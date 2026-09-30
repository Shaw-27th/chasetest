"""N日动量指数轮动：按横截面动量排名生成目标仓位。"""

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
    previous_codes: list[str] = []

    for trade_date, factor_values in factor_panel.iterrows():
        valid_values = factor_values.dropna()
        if valid_values.empty:
            previous_codes = []
            continue

        best_value = valid_values.min() if factor.is_sort_asc else valid_values.max()
        if params.get("empty_when_all_negative", False) and best_value < 0:
            previous_codes = []
            continue

        select_num = min(max_select_num, len(valid_values))
        ranked_codes = valid_values.sort_values(
            ascending=factor.is_sort_asc, kind="stable"
        ).index.tolist()
        cutoff_value = valid_values.loc[ranked_codes[select_num - 1]]
        better_codes = [
            code
            for code in ranked_codes
            if (valid_values.loc[code] < cutoff_value if factor.is_sort_asc else valid_values.loc[code] > cutoff_value)
        ]
        tied_codes = [code for code in ranked_codes if valid_values.loc[code] == cutoff_value]
        remaining_num = select_num - len(better_codes)

        if params.get("tie_break", "order") == "hold":
            held_ties = [code for code in previous_codes if code in tied_codes]
            new_ties = [code for code in tied_codes if code not in held_ties]
            selected_codes = better_codes + (held_ties + new_ties)[:remaining_num]
        else:
            selected_codes = better_codes + tied_codes[:remaining_num]

        ratios.loc[trade_date, selected_codes] = 1.0 / len(selected_codes)
        previous_codes = selected_codes

    return ratios
#󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
