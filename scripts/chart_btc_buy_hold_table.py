"""
Exhibit -- BTC buy & hold metrics table, Jan 2022 to Aug 2026.

Reads data/processed/btc_buy_hold_2022_2026-08.csv (scripts/build_btc_buy_hold.py).
Metric definitions come from src/metrics.py only.
"""
import os
import sys

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from house_style import FONT, BLACK, TABLE_ROW  # noqa: E402
from src import metrics  # noqa: E402

OUT = "charts/btc_buy_hold/final"
os.makedirs(OUT, exist_ok=True)

bt = pd.read_csv("data/processed/btc_buy_hold_2022_2026-08.csv", index_col=0, parse_dates=True)
table = metrics.format_summary(metrics.summary(bt, metrics.PPY_CRYPTO))

n = len(table)
fig = go.Figure(go.Table(
    columnwidth=[1.4, 1],
    header=dict(values=["<b>Metric</b>", "<b>BTC buy & hold</b>"], fill_color=BLACK, align="left",
                font=dict(family=FONT, size=16, color="white"), height=44),
    cells=dict(values=[table.index.tolist(), table.tolist()], align="left", height=36,
               fill_color=[["white" if i % 2 == 0 else TABLE_ROW for i in range(n)]],
               font=dict(family=FONT, size=15, color="black"), line_color="white"),
))
fig.update_layout(width=900, height=40 + 44 + 36 * n + 110, paper_bgcolor="white",
                  margin=dict(l=20, r=20, t=20, b=90))
fig.add_annotation(
    text=("Total return and CAGR are geometrically compounded. Gross of transaction costs, slippage and fees.<br>"
          "Source: Coinbase BTC-USD daily closes (00:00 UTC), 31 December 2021 to 31 August 2026."),
    xref="paper", yref="paper", x=0, y=0, yshift=-70, showarrow=False, xanchor="left", align="left",
    font=dict(family=FONT, size=13, color="grey"),
)

path = f"{OUT}/btc_buy_hold_table.png"
fig.write_image(path, scale=2)
print(f"saved {path}")
print(table.to_string())
