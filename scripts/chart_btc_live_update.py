"""
Exhibit -- tracking update: growth of $1 for every BTC strategy from 31 Dec 2021 to the latest cached bar
(log scale). The dashed line marks the end of the pre-registered window (31 Aug 2026); everything to its
right is out of sample for every spec.

Reads data/processed/btc_live_update.csv (scripts/build_btc_live_update.py).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from house_style import create_default_figure, BLUE, DGREY, GREEN, LGREY  # noqa: E402

OUT = "charts/btc_live_update/final"
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv("data/processed/btc_live_update.csv", index_col=0, parse_dates=True)
COLORS = {"BTC buy & hold": DGREY, "50/200 SMA cross L/S": LGREY,
          "Close vs SMA 50 L/S (Study A)": BLUE, "12m mom + 200d SMA L/S": GREEN}
last = d.index[-1]

fig = create_default_figure(
    ylabel="Growth of $1 (log scale)",
    source_text=(f"Coinbase BTC-USD daily closes (00:00 UTC), $1 at the close of 31 December 2021, marked to the close of<br>"
                 f"{last:%d %B %Y}. Dashed line: end of the pre-registered window (31 August 2026); to its right is out of<br>"
                 "sample for every rule, unchanged. Long/short with shorts held; set at each close, held from the next. Gross."),
    ysource_text_offset=0.40,
    ylegend_offset=0.22,
    width=1000,
    height=620,
)
for name, color in COLORS.items():
    tot = d[name].iloc[-1] - 1
    fig.add_scatter(x=d.index, y=d[name], mode="lines", name=f"{name}  ({tot:+,.0%})".replace("-", "−"),
                    line=dict(color=color, width=2.5))
cut = pd.Timestamp("2026-08-31", tz=d.index.tz)
fig.add_vline(x=cut, line=dict(color="black", dash="dash", width=1.2))

lo, hi = d.min().min(), d.max().max()
ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8) if lo * 0.85 <= t <= hi * 1.15]
fig.update_layout(margin=dict(l=100, r=50, t=30, b=190), showlegend=True)
fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"${t:g}" for t in ticks],
                 range=[np.log10(lo * 0.9), np.log10(hi * 1.1)])
fig.update_xaxes(dtick="M12", tickformat="%Y", range=[d.index[0], last])

path = f"{OUT}/btc_live_update_growth.png"
fig.write_image(path, scale=2)
print(f"saved {path}")
