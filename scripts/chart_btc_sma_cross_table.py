"""
Exhibit -- metrics table: BTC 50/200-day SMA long/short vs BTC buy & hold, Jan 2022 to Aug 2026.

Reads data/processed/btc_sma_cross_50_200_2022_2026-08.csv (scripts/build_btc_sma_cross.py)
and data/processed/btc_buy_hold_2022_2026-08.csv (scripts/build_btc_buy_hold.py).
Metric definitions come from src/metrics.py only.
"""
import os
import sys

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from house_style import FONT, BLACK, TABLE_BLUE, TABLE_GREEN, TABLE_ROW  # noqa: E402
from src import metrics  # noqa: E402

OUT = "charts/btc_sma_cross_50_200/final"
os.makedirs(OUT, exist_ok=True)

strat = pd.read_csv("data/processed/btc_sma_cross_50_200_2022_2026-08.csv", index_col=0, parse_dates=True)
bench = pd.read_csv("data/processed/btc_buy_hold_2022_2026-08.csv", index_col=0, parse_dates=True)
assert strat.index.equals(bench.index)
b = metrics.format_summary(metrics.summary(bench, metrics.PPY_CRYPTO))
s = metrics.format_summary(metrics.summary(strat, metrics.PPY_CRYPTO))

n = len(b)
stripe = ["white" if i % 2 == 0 else TABLE_ROW for i in range(n)]
fig = go.Figure(go.Table(
    columnwidth=[1.4, 1, 1.1],
    header=dict(values=["<b>Metric</b>", "<b>BTC buy & hold</b>", "<b>50/200 SMA long/short</b>"],
                fill_color=[BLACK, BLACK, TABLE_BLUE], align="left",
                font=dict(family=FONT, size=16, color="white"), height=44),
    cells=dict(values=[b.index.tolist(), [v.replace("-", "−") for v in b.tolist()], [v.replace("-", "−") for v in s.tolist()]],
               align="left", height=36,
               fill_color=[stripe, stripe, [TABLE_GREEN] * n],
               font=dict(family=FONT, size=15, color="black"), line_color="white"),
))
fig.update_layout(width=1000, height=40 + 44 + 36 * n + 130, paper_bgcolor="white",
                  margin=dict(l=20, r=20, t=20, b=110))
fig.add_annotation(
    text=("Strategy: long BTC when the 50-day SMA is above the 200-day, short (held) when below; set at each close, held from the next.<br>"
          "Total return and CAGR are geometrically compounded. Gross of transaction costs, slippage, fees and short funding.<br>"
          "Source: Coinbase BTC-USD daily closes (00:00 UTC), 31 December 2021 to 31 August 2026."),
    xref="paper", yref="paper", x=0, y=0, yshift=-85, showarrow=False, xanchor="left", align="left",
    font=dict(family=FONT, size=13, color="grey"),
)

path = f"{OUT}/btc_sma_cross_table.png"
fig.write_image(path, scale=2)
print(f"saved {path}")
print(pd.DataFrame({"BTC buy & hold": b, "50/200 L/S": s}).to_string())
