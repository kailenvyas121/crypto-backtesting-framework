"""
Exhibit -- BTC buy & hold drawdown from peak, Jan 2022 to Aug 2026.

Reads data/processed/btc_buy_hold_2022_2026-08.csv (scripts/build_btc_buy_hold.py).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from house_style import create_default_figure, DGREY, SHADE  # noqa: E402

OUT = "charts/btc_buy_hold/final"
os.makedirs(OUT, exist_ok=True)

bt = pd.read_csv("data/processed/btc_buy_hold_2022_2026-08.csv", index_col=0, parse_dates=True)
trough = bt.drawdown.min()

fig = create_default_figure(
    ylabel="Drawdown from peak",
    source_text=("Coinbase BTC-USD daily closes (00:00 UTC), 31 December 2021 to 31 August 2026. "
                 "Peak includes the<br>entry value. Gross of transaction costs and fees."),
    ysource_text_offset=0.30,
    ylegend_offset=0.20,
    width=1000,
    height=560,
)
fig.add_scatter(x=bt.index, y=bt.drawdown, mode="lines", fill="tozeroy", fillcolor=SHADE,
                name=f"BTC buy & hold  ({trough:.0%} trough)".replace("-", "−"),
                line=dict(color=DGREY, width=1.5))

fig.update_layout(margin=dict(l=100, r=50, t=30, b=140), showlegend=True)
fig.update_yaxes(tickformat=".0%", dtick=0.1, range=[min(trough * 1.08, -0.1), 0.02])
fig.update_xaxes(dtick="M12", tickformat="%Y", range=[bt.index[0], bt.index[-1]])

path = f"{OUT}/btc_buy_hold_drawdown.png"
fig.write_image(path, scale=2)
print(f"saved {path}")
print(f"max drawdown {trough:.1%} on {bt.drawdown.idxmin().date()}   "
      f"prior peak {bt.equity.loc[:bt.drawdown.idxmin()].idxmax().date()}")
