"""
MA10均线择时 回测示例。

读取 xlsx 日线数据，执行 MA10 择时，输出持仓序列。

文件路径：D:\stock_data\stock-0MVA-data-2026-08-09\0MVA.xlsx
列名：日期、开盘、最高、最低、收盘、成交量、成交额
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 路径配置
XLSX_PATH = r"D:\stock_data\stock-0MVA-data-2026-08-09\0MVA.xlsx"

# MA10择时参数
MODE = "cross"      # "cross"（默认）金叉买死叉卖；"state" 逐日判断
CONFIRM_N = 1       # 条件需连续成立 n 天才确认


def timing_signal_from_df(stock_df: pd.DataFrame, mode: str = "cross", confirm_n: int = 1) -> pd.Series:
    """
    直接对 DataFrame 执行 MA10 择时。
    stock_df 必须包含 收盘 列。
    """
    close = stock_df["收盘"]

    # 计算 MA10
    ma10 = close.rolling(window=10, min_periods=1).mean()

    # 多空原始条件
    long_con  = close > ma10   # 收盘价在 MA10 上方 -> 持有
    short_con = close < ma10   # 收盘价在 MA10 下方 -> 空仓

    # 预热期不产生条件
    valid = close.notna()
    long_con  &= valid
    short_con &= valid

    if confirm_n > 1:
        long_con  = long_con.rolling(confirm_n).sum().eq(confirm_n)
        short_con = short_con.rolling(confirm_n).sum().eq(confirm_n)

    if mode == "state":
        return long_con.astype("float64").rename("持仓状态")

    # cross：只在条件首次成立那天产生买卖事件，其余日期维持原持仓
    signal = pd.Series(np.nan, index=stock_df.index)
    signal[long_con  & ~long_con.shift(1, fill_value=False)] = 1.0   # 金叉 -> 买入
    signal[short_con & ~short_con.shift(1, fill_value=False)] = 0.0  # 死叉 -> 卖出
    return signal.ffill().fillna(0.0).rename("持仓状态")


def main():
    print("[Step1] Read data: " + XLSX_PATH)
    df = pd.read_excel(XLSX_PATH)

    # 确保日期列是datetime类型并设为索引
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    df = df.set_index("日期")

    print(f"[OK] Data loaded: {len(df)} rows, {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"     Columns: {df.columns.tolist()}")

    # 计算持仓状态
    signal = timing_signal_from_df(df, mode=MODE, confirm_n=CONFIRM_N)
    df["持仓状态"] = signal

    # 简单统计
    hold_days = int((df["持仓状态"] == 1).sum())
    total_days = len(df)
    print(f"\n[Stats] Timing result (mode={MODE}, confirm_n={CONFIRM_N})")
    print(f"  Hold days:   {hold_days} / {total_days} ({hold_days/total_days*100:.1f}%)")
    print(f"  Empty days:  {total_days - hold_days} / {total_days} ({(total_days-hold_days)/total_days*100:.1f}%)")

    # 展示买卖点
    change_points = df["持仓状态"].diff().fillna(0).abs() > 0
    if change_points.any():
        trades = df[change_points].copy()
        print(f"\n[Trade] {change_points.sum()} trade points:")
        for date, row in trades.iterrows():
            action = "BUY" if row["持仓状态"] == 1 else "SELL"
            close_price = row["收盘"]
            print(f"  {date.date()}  {action}  close={close_price:.2f}")

    # ===== 实时状态显示 =====
    print("\n" + "="*60)
    print("【实时状态】最近5个交易日：")
    print("="*60)
    
    # 计算MA10用于显示
    ma10 = df["收盘"].rolling(window=10, min_periods=1).mean()
    
    recent = df[["收盘", "持仓状态"]].tail(5).copy()
    recent["MA10"] = ma10.tail(5)
    
    for date, row in recent.iterrows():
        status = "✅ 在MA10上方（持仓）" if row["持仓状态"] == 1 else "❌ 在MA10下方（空仓）"
        print(f"  {date.strftime('%Y-%m-%d')}  收盘:{row['收盘']:>10.2f}  MA10:{row['MA10']:>10.2f}  {status}")
    
    # 当前最新状态
    latest = df.iloc[-1]
    latest_date = df.index[-1]
    latest_close = latest["收盘"]
    latest_ma10 = ma10.iloc[-1]
    latest_status = "✅ 持仓中" if latest["持仓状态"] == 1 else "❌ 空仓中"
    
    print("\n" + "="*60)
    print("【当前最新】")
    print("="*60)
    print(f"  日期:     {latest_date.strftime('%Y-%m-%d')}")
    print(f"  收盘价:   {latest_close:.2f}")
    print(f"  MA10:    {latest_ma10:.2f}")
    print(f"  状态:     {latest_status}")
    print(f"  差值:     {latest_close - latest_ma10:+.2f} ({((latest_close/latest_ma10)-1)*100:+.2f}%)")
    print("="*60)

    # 保存结果
    out_path = XLSX_PATH.replace(".xlsx", "_MA10_result.xlsx")
    df[["收盘", "持仓状态"]].to_excel(out_path)
    print(f"\n[Saved] Result: {out_path}")


if __name__ == "__main__":
    main()
