"""
Exhibit -- growth of $1 in BTC buy & hold, Jan 2022 to Aug 2026 (log scale).

Reads data/processed/btc_buy_hold_2022_2026-08.csv (scripts/build_btc_buy_hold.py).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from house_style import create_default_figure, DGREY  # noqa: E402

OUT = "charts/btc_buy_hold/final"
os.makedirs(OUT, exist_ok=True)

bt = pd.read_csv("data/processed/btc_buy_hold_2022_2026-08.csv", index_col=0, parse_dates=True)
total = bt.equity.iloc[-1] - 1

fig = create_default_figure(
    ylabel="Growth of $1 (log scale)",
    source_text=("Coinbase BTC-USD daily closes (00:00 UTC). $1 invested at the close of 31 December 2021, "
                 "marked to the<br>close of 31 August 2026. Gross of transaction costs and fees."),
    ysource_text_offset=0.30,
    ylegend_offset=0.20,
    width=1000,
    height=560,
)
fig.add_scatter(x=bt.index, y=bt.equity, mode="lines", name=f"BTC buy & hold  ({total:+,.0%})",
                line=dict(color=DGREY, width=2.5))

ticks = [0.25, 0.5, 1, 2, 3]
fig.update_layout(margin=dict(l=100, r=50, t=30, b=140), showlegend=True)
fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"${t:g}" for t in ticks],
                 range=[np.log10(bt.equity.min() * 0.9), np.log10(bt.equity.max() * 1.1)])
fig.update_xaxes(dtick="M12", tickformat="%Y", range=[bt.index[0], bt.index[-1]])

path = f"{OUT}/btc_buy_hold_growth.png"
fig.write_image(path, scale=2)
print(f"saved {path}")
print(f"total return {total:+.1%}   final value ${bt.equity.iloc[-1]:.3f}   "
      f"low ${bt.equity.min():.3f} on {bt.equity.idxmin().date()}   high ${bt.equity.max():.3f} on {bt.equity.idxmax().date()}")
