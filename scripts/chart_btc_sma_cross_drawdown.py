"""
Exhibit -- drawdown from peak: BTC 50/200-day SMA long/short vs BTC buy & hold, Jan 2022 to Aug 2026.

Reads data/processed/btc_sma_cross_50_200_2022_2026-08.csv (scripts/build_btc_sma_cross.py)
and data/processed/btc_buy_hold_2022_2026-08.csv (scripts/build_btc_buy_hold.py).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from house_style import create_default_figure, DGREY, GREEN, SHADE  # noqa: E402

OUT = "charts/btc_sma_cross_50_200/final"
os.makedirs(OUT, exist_ok=True)

strat = pd.read_csv("data/processed/btc_sma_cross_50_200_2022_2026-08.csv", index_col=0, parse_dates=True)
bench = pd.read_csv("data/processed/btc_buy_hold_2022_2026-08.csv", index_col=0, parse_dates=True)
assert strat.index.equals(bench.index)
s_dd, b_dd = strat.drawdown.min(), bench.drawdown.min()

fig = create_default_figure(
    ylabel="Drawdown from peak",
    source_text=("Coinbase BTC-USD daily closes (00:00 UTC), 31 December 2021 to 31 August 2026. Peak includes the<br>"
                 "entry value. Strategy: long when 50-day SMA > 200-day, short (held) when below. Gross of costs,<br>"
                 "fees and short funding."),
    ysource_text_offset=0.36,
    ylegend_offset=0.20,
    width=1000,
    height=600,
)
fig.add_scatter(x=bench.index, y=bench.drawdown, mode="lines", fill="tozeroy", fillcolor=SHADE,
                name=f"BTC buy & hold  ({b_dd:.0%} trough)".replace("-", "−"),
                line=dict(color=DGREY, width=1.5))
fig.add_scatter(x=strat.index, y=strat.drawdown, mode="lines",
                name=f"50/200 SMA long/short  ({s_dd:.0%} trough)".replace("-", "−"),
                line=dict(color=GREEN, width=2))

floor = min(s_dd, b_dd)
fig.update_layout(margin=dict(l=100, r=50, t=30, b=170), showlegend=True)
fig.update_yaxes(tickformat=".0%", dtick=0.1, range=[min(floor * 1.08, -0.1), 0.02])
fig.update_xaxes(dtick="M12", tickformat="%Y", range=[strat.index[0], strat.index[-1]])

path = f"{OUT}/btc_sma_cross_drawdown.png"
fig.write_image(path, scale=2)
print(f"saved {path}")
for name, d in (("strategy", strat), ("buy & hold", bench)):
    t = d.drawdown.idxmin()
    print(f"{name:11s} max drawdown {d.drawdown.min():.1%} on {t.date()}   prior peak {d.equity.loc[:t].idxmax().date()}")
