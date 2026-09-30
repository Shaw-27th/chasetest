"""
邢不行™️选股框架
Python股票量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import importlib

import pandas as pd


class FactorInterface:
    """
    ！！！！抽象因子对象，仅用于代码提示！！！！
    """

    # 财务因子列：此列表用于存储财务因子相关的列名称󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    fin_cols = []  # 财务因子列，配置后系统会自动加载对应的财务数据󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿

    @staticmethod
    def add_factor(df: pd.DataFrame, param=None, **kwargs) -> pd.DataFrame:
        """
        计算因子，并返回包含新因子列的 DataFrame。

        工作流程：
        1. 根据提供的参数计算股票的因子值。
        2. 将因子值放入以 ``col_name`` 命名的列中。
        3. 只返回因子列，框架负责与公共日线 K 线按行对齐和缓存。

        :param df: pd.DataFrame，包含单只股票的K线数据，必须包括市场数据（如收盘价等）。
        :param param: 因子计算所需的参数，格式和含义根据因子类型的不同而有所不同。
        :param kwargs: 其他关键字参数，包括：
            - col_name: 新计算的因子列名。
            - fin_data: 财务数据字典，格式为 {'财务数据': fin_df, '原始财务数据': raw_fin_df}，其中fin_df为处理后的财务数据，raw_fin_df为原始数据，后者可用于某些因子的自定义计算。
            - 其他参数：根据具体需求传入的其他因子参数。
        :return: 包含新计算因子列的 DataFrame，与输入 df 的索引一致。

        注意事项：
        - 因子计算不得增删或重排数据行，返回结果必须与输入 df 的行数和索引一致。
        - 涉及财务数据时，通过 ``fin_cols`` 声明所需字段，并可从 ``fin_data`` 读取原始财报数据。
        - 因子模块不再声明周期聚合方式，换仓日期由选股阶段统一处理。
        """

        # ======================== 参数处理 ===========================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        # 从kwargs中提取因子列的名称，这里使用'col_name'来标识因子列名称󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        col_name = kwargs["col_name"]

        # ======================== 计算因子 ===========================󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
        """
        [abstract]
        目前这个接口中并没有实现任何的计算逻辑，只是提供一个接口，用于提示
        需要在这个位置实现计算逻辑，并且在 `df` 中添加一个新的因子列，列名为 col_name
        """

        return df[[col_name]]

    def add_factors(self, df: pd.DataFrame, params=(), **kwargs) -> pd.DataFrame:
        """
        批量计算多个参数下的因子数值。

        :param df: 单只股票的日线数据。
        :param params: 同一因子需要计算的参数集合。
        :param kwargs: 传递给具体因子实现的其他参数。
        :return: 包含各参数对应因子列的 DataFrame，行数和索引与输入 df 一致。
        """
        raise NotImplementedError


class FactorHub:
    _factor_cache = {}

    # noinspection PyTypeChecker󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
    @staticmethod
    def get_by_name(factor_name) -> FactorInterface:
        if factor_name in FactorHub._factor_cache:
            return FactorHub._factor_cache[factor_name]

        try:
            # 构造模块名󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            module_name = f"因子库.{factor_name}"

            # 动态导入模块󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            factor_module = importlib.import_module(module_name)

            # 创建一个包含模块变量和函数的字典󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            factor_content = {
                name: getattr(factor_module, name) for name in dir(factor_module) if not name.startswith("__")
            }

            if "fin_cols" not in factor_content:
                factor_content["fin_cols"] = []

            # 创建一个包含这些变量和函数的对象󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            factor_instance = type(factor_name, (), factor_content)

            # 缓存策略对象󠀂󠁥󠀵󠁡󠁥󠀸󠀷󠀲󠀰󠀳󠀱󠀳󠀷󠀳󠀹󠀳󠀷󠀳󠀷󠀳󠀰󠁿
            FactorHub._factor_cache[factor_name] = factor_instance

            return factor_instance
        except ModuleNotFoundError:
            raise ValueError(f"Factor {factor_name} not found.")
        except AttributeError:
            raise ValueError(f"Error accessing factor content in module {factor_name}.")
