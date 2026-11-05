"""
Exhibit -- Study A: Newey-West t-stat of each trend feature's next-day return spread, discovery
(Jul 2016 - Dec 2021) vs holdout (Jan 2022 - Aug 2026), in discovery-rank order. The dashed line is
the 95th percentile of the shuffled best-of-13 t-stat.

Reads data/processed/btc_trend_predictability.csv and _null.csv (scripts/build_btc_trend_predictability.py).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from house_style import create_default_figure, DGREY, GREEN  # noqa: E402

OUT = "charts/btc_trend_predictability/final"
os.makedirs(OUT, exist_ok=True)

res = pd.read_csv("data/processed/btc_trend_predictability.csv").query("horizon == 1")
null = pd.read_csv("data/processed/btc_trend_predictability_null.csv")
p95 = null.discovery.quantile(0.95)
disc = res.query("period == 'discovery'").sort_values("discovery_rank")
hold = res.query("period == 'holdout'").set_index("feature").loc[disc.feature]

LABELS = {"mom_30": "30d return > 0", "mom_90": "90d return > 0", "mom_180": "180d return > 0",
          "mom_365": "365d return > 0", "px_sma_20": "Price > SMA 20", "px_sma_50": "Price > SMA 50",
          "px_sma_100": "Price > SMA 100", "px_sma_200": "Price > SMA 200", "cross_20_50": "SMA 20 > 50",
          "cross_50_100": "SMA 50 > 100", "cross_50_200": "SMA 50 > 200", "cross_100_200": "SMA 100 > 200",
          "mom365_sma200": "Spec 3 rule"}
x = [LABELS[f] for f in disc.feature]

fig = create_default_figure(
    ylabel="t-stat of next-day return spread",
    source_text=("Coinbase BTC-USD daily. Spread = mean next-day return when the feature is on minus when off; Newey-West<br>"
                 "t-stat (10 lags). Features ranked on discovery (Jul 2016 - Dec 2021) only; holdout (Jan 2022 - Aug 2026)<br>"
                 "shown in the same order, never re-ranked. Dashed: 95th pct of best-of-13 t under 1,000 circular shuffles."),
    ysource_text_offset=0.52,
    ylegend_offset=0.40,
    width=1000,
    height=650,
)
fig.add_bar(x=x, y=disc.t_stat, name="Discovery 2016-2021", marker_color=DGREY)
fig.add_bar(x=x, y=hold.t_stat, name="Holdout 2022-2026", marker_color=GREEN)
fig.add_hline(y=p95, line=dict(color="black", dash="dash", width=1.5))
fig.add_annotation(x=x[-1], y=p95, text=f"best-of-13 null 95%: {p95:.2f}", showarrow=False,
                   xanchor="right", yanchor="bottom", font=dict(size=13))
fig.update_layout(barmode="group", margin=dict(l=100, r=50, t=30, b=260), showlegend=True)
fig.update_xaxes(tickangle=-40)

path = f"{OUT}/btc_trend_predictability_tstats.png"
fig.write_image(path, scale=2)
print(f"saved {path}")
