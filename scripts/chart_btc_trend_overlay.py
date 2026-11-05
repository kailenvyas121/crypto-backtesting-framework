"""
Exhibit -- growth of $1 on the holdout window, Jan 2022 to Aug 2026 (log scale): buy & hold, the 50/200
cross (spec 2), the Study A discovery winner and the spec 3 rule, all long/short with held shorts.

Reads the processed backtests from scripts/build_btc_buy_hold.py, build_btc_sma_cross.py,
build_btc_trend_predictability.py and build_btc_mom_sma.py.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from house_style import create_default_figure, BLUE, DGREY, GREEN, LGREY  # noqa: E402

OUT = "charts/btc_trend_predictability/final"
os.makedirs(OUT, exist_ok=True)

P = "data/processed/"
runs = [
    ("BTC buy & hold", "btc_buy_hold_2022_2026-08.csv", DGREY),
    ("50/200 SMA cross L/S", "btc_sma_cross_50_200_2022_2026-08.csv", LGREY),
    ("Study A winner: price vs SMA 50 L/S", "btc_trend_winner_2022_2026-08.csv", BLUE),
    ("12m mom + 200d SMA L/S", "btc_mom_sma_2022_2026-08.csv", GREEN),
]
frames = {name: pd.read_csv(P + f, index_col=0, parse_dates=True) for name, f, _ in runs}
rank1 = pd.read_csv(P + "btc_trend_predictability.csv").query("horizon == 1 and discovery_rank == 1").feature.iloc[0]
assert rank1 == "px_sma_50", f"Study A winner is now {rank1}: update the label and the builder's overlay"
idx = frames["BTC buy & hold"].index
assert all(d.index.equals(idx) for d in frames.values())

fig = create_default_figure(
    ylabel="Growth of $1 (log scale)",
    source_text=("Coinbase BTC-USD daily closes (00:00 UTC). $1 at the close of 31 December 2021, marked to 31 August 2026.<br>"
                 "All strategies long/short, shorts held; position set at each close, held from the next. The Study A winner<br>"
                 "was chosen on 2016-2021 data only. Gross of costs, fees and short funding."),
    ysource_text_offset=0.40,
    ylegend_offset=0.22,
    width=1000,
    height=620,
)
for name, _, color in runs:
    d = frames[name]
    tot = d.equity.iloc[-1] - 1
    fig.add_scatter(x=d.index, y=d.equity, mode="lines", name=f"{name}  ({tot:+,.0%})".replace("-", "−"),
                    line=dict(color=color, width=2.5))

lo = min(d.equity.min() for d in frames.values())
hi = max(d.equity.max() for d in frames.values())
ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8) if lo * 0.85 <= t <= hi * 1.15]
fig.update_layout(margin=dict(l=100, r=50, t=30, b=190), showlegend=True)
fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"${t:g}" for t in ticks],
                 range=[np.log10(lo * 0.9), np.log10(hi * 1.1)])
fig.update_xaxes(dtick="M12", tickformat="%Y", range=[idx[0], idx[-1]])

path = f"{OUT}/btc_trend_overlay_growth.png"
fig.write_image(path, scale=2)
print(f"saved {path}")
