"""
Exhibit -- growth of $1: BTC 50/200-day SMA long/short vs BTC buy & hold, Jan 2022 to Aug 2026 (log scale).
Shaded bands mark the periods the strategy was short.

Reads data/processed/btc_sma_cross_50_200_2022_2026-08.csv (scripts/build_btc_sma_cross.py)
and data/processed/btc_buy_hold_2022_2026-08.csv (scripts/build_btc_buy_hold.py).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from house_style import create_default_figure, DGREY, GREEN, SHADE  # noqa: E402

OUT = "charts/btc_sma_cross_50_200/final"
os.makedirs(OUT, exist_ok=True)

strat = pd.read_csv("data/processed/btc_sma_cross_50_200_2022_2026-08.csv", index_col=0, parse_dates=True)
bench = pd.read_csv("data/processed/btc_buy_hold_2022_2026-08.csv", index_col=0, parse_dates=True)
assert strat.index.equals(bench.index)
s_tot, b_tot = strat.equity.iloc[-1] - 1, bench.equity.iloc[-1] - 1

fig = create_default_figure(
    ylabel="Growth of $1 (log scale)",
    source_text=("Coinbase BTC-USD daily closes (00:00 UTC). $1 invested at the close of 31 December 2021, marked to the<br>"
                 "close of 31 August 2026. Long when the 50-day SMA is above the 200-day, short (held) when below; shaded<br>"
                 "= short. Position set at each close, held from the next. Gross of costs, fees and short funding."),
    ysource_text_offset=0.36,
    ylegend_offset=0.20,
    width=1000,
    height=600,
)

short = strat.position.lt(0)
starts = strat.index[short & ~short.shift(1, fill_value=False)]
ends = strat.index[short & ~short.shift(-1, fill_value=False)]
for a, b in zip(starts, ends):
    fig.add_vrect(x0=a - pd.Timedelta(days=1), x1=b, fillcolor=SHADE, opacity=0.6, line_width=0, layer="below")

fig.add_scatter(x=bench.index, y=bench.equity, mode="lines", name=f"BTC buy & hold  ({b_tot:+,.0%})",
                line=dict(color=DGREY, width=2.5))
fig.add_scatter(x=strat.index, y=strat.equity, mode="lines",
                name=f"50/200 SMA long/short  ({s_tot:+,.0%})".replace("-", "−"),
                line=dict(color=GREEN, width=2.5))

lo = min(strat.equity.min(), bench.equity.min())
hi = max(strat.equity.max(), bench.equity.max())
ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4) if lo * 0.85 <= t <= hi * 1.15]
fig.update_layout(margin=dict(l=100, r=50, t=30, b=170), showlegend=True)
fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"${t:g}" for t in ticks],
                 range=[np.log10(lo * 0.9), np.log10(hi * 1.1)])
fig.update_xaxes(dtick="M12", tickformat="%Y", range=[strat.index[0], strat.index[-1]])

path = f"{OUT}/btc_sma_cross_growth.png"
fig.write_image(path, scale=2)
print(f"saved {path}")
print(f"strategy {s_tot:+.1%}   buy & hold {b_tot:+.1%}   short periods: {len(starts)}")
