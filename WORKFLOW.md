# Research Workflow

How this project is set up and run. Written to be copied into the next one — the
folder layout, the conventions, and the handful of non-obvious things that cost
time to discover the first time.

---

## The shape of it

Three tiers, and the split is the whole trick:

| Tier | What lives there | Rewritten? |
|---|---|---|
| `notebooks/*.ipynb` | Exploration. Markdown reasoning between cells. Messy by design. | Constantly |
| `src/*.py` | Any function called more than once. Data pulls, metric definitions, stats. | Rarely |
| `scripts/*.py` | One-shot runnables. Final chart renderers, verification, data builders. | Frozen once correct |
The notebook is where you think; it is not the deliverable and nothing final
should only exist there. The moment a computation matters, it moves into `src/`
so both the notebook and a script can call the identical code. The moment a
chart is final, it becomes a script that regenerates it from disk in one command.
Why not do the charts in the notebook: a notebook cell's output is a picture of
state you can no longer inspect. Six weeks later you cannot tell whether the PNG
matches the code above it. A script you can just re-run.

## Folder layout

```
project/
├── README.md              # thesis, pre-registered spec, universe, standards
├── data_map.md            # provenance: every source, date range, row count, caveat
├── WORKFLOW.md            # this file
├── requirements.txt       # pinned
├── .gitignore             # .venv/ __pycache__/ .ipynb_checkpoints/ data/raw/ .env
│
├── notebooks/             # exploration
├── src/                   # library: data.py, analysis.py, charts.py
├── scripts/               # runnables: build_*.py, chart scripts, verify_*.py
│
├── data/
│   ├── raw/               # gitignored — API pulls, never edited by hand
│   ├── processed/         # committed — every file has a named builder script
│   └── findings_log.csv   # append-only, one row per run, nulls included
│
├── charts/
│   └── <phase>/final/     # the deliverable set
│       └── archive/       # superseded, never deleted
│
└── report/                # the write-ups
```
## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pin everything. The one pin that actually matters:

```
kaleido==0.2.1    # PNG export for Plotly. v1 hangs forever trying to fetch Chrome.
```

If `fig.write_image()` ever hangs or dies with no traceback, check the kaleido
version before you debug anything else. This burned an afternoon.

Kaleido 0.2.1 also segfaults (`SIGSEGV`, no useful traceback) when run inside a
restricted sandbox, including Cursor's default agent shell. The code is fine —
the Chrome subprocess it spawns is not allowed to start. Run chart scripts in a
normal terminal, or grant the agent full permissions, before debugging Plotly.

Run scripts from the project root, always, so relative paths resolve:

```bash
.venv/bin/python scripts/micro_dose_response.py
```
## The notebook bootstrap cell

First code cell of every notebook, unchanged. It makes the notebook work
regardless of whether Jupyter's cwd is the project root or `notebooks/`, and
`importlib.reload` means you can edit `src/` in Cursor and pick up the change
without restarting the kernel.

```python
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from src import micro_data as md
from src import micro_analysis as ma

import importlib
md = importlib.reload(md)
ma = importlib.reload(ma)

# constants defining the spec live here, at the top, named
CUTOFFS = (60, 120, 300) # seconds before close
HORIZONS = (1, 5, 15, 30) # minutes after close
NEAR_5050_BAND = 0.15

pd.set_option("display.width", 160)
```

Constants at the top rather than inline is what let the same spec be reused
across six assets and a dozen scripts without drift.
## The loop

**1. Write down the spec before running it.** A short section in `README.md`
saying what you are testing, on what universe, with what controls, and what
would count as a null. This is the only defence against deciding after the fact
that whatever came back was what you were looking for.

**2. Explore in the notebook.** Markdown cell stating the question, code cell
answering it, markdown cell recording what came back — including when it is
nothing. Those markdown cells become the write-up later.

**3. Log every run.** Append to `data/findings_log.csv` as you go:

```
date_logged,spec,asset,cutoff_s,horizon_min,notes
2026-08-11,concurrent_move_control,BTC,60,5,"uni_beta=0.00012, uni_p=0.13459, survives_control=False"
```

268 rows in this project. This is the single highest-value habit here. It is
what made the "what came back null" section writable in an afternoon instead of
a week, and it is what stops you re-running a dead end you already killed in
July. Log the failures especially — they are most of the file and most of the
credibility.

**4. Promote to `src/`** once a function is called twice. Put the design rules
in the module docstring, not in your head:

> No lookahead, ASSERTED not just assumed: every forward metric uses only bars
> whose OPEN time is >= the market's close_ts. Both are checked in code, not
> just by convention.

**5. Document provenance in `data_map.md`.** Every processed file gets a section
with its exact source, date range, row count, and known caveats. If a CSV in
`data/processed/` has no builder script and no data_map entry, it is orphaned —
you cannot regenerate it and you cannot trust it. Two files here reached the
final chart set in that state and both had to be reverse-engineered into
`build_*.py` scripts at the end. Do it as you go instead.

**6. Freeze the chart as a script.** One script per chart, reading from
`data/processed/`, importing `house_style.py`, ending in
`fig.write_image(path, scale=2)`. Have it print its headline numbers to stdout
after saving, so the figure and the prose you write about it come from one run.

**7. Verify the prose against the data.** `scripts/verify_note_numbers.py`
recomputes every number quoted in the write-up and asserts it matches. Every
figure in the note, not the interesting ones. This caught a volatility metric
quoted as a standard deviation in one paragraph and a mean absolute return in
another — same word, different number, and no reader would have known which.

**8. Check for staleness before shipping.**

```bash
python scripts/check_freshness.py # verify
python scripts/check_freshness.py --update # re-record after re-rendering
```

A chart is stale when its inputs changed after it was rendered, and a stale
chart is worse than a missing one because it looks finished. The obvious check
is mtimes — chart newer than every script and CSV it reads — and it does not
work. Every script in this repo shares an identical mtime to the second from a
bulk touch that changed no content, which makes the entire deck look stale at
once and trains you to ignore the check. `check_freshness.py` hashes each
chart's declared inputs against a recorded manifest instead, so it only fires on
real drift. The one manual step is adding to a chart's dependency list when it
gains an input.

Worth knowing: these renders are byte-reproducible. Re-running a chart script
with unchanged inputs produces an identical PNG, so if a hash does move,
something real moved.

**9. Archive, never delete.** Superseded charts go to `charts/<phase>/final/archive/`.
Fourteen of them here. The final folder should contain exactly the deliverable
set so there is no ambiguity about which PNG is current, but the dead ones stay
around because "why did you drop the box plot" is a question that gets asked.

---

## Charting

All the plot code and the reusable Plotly patterns are in the previous message
and in `scripts/house_style.py`. The workflow points:

- One `house_style.py` with `create_default_figure()` and the palette as module
constants. Restyling the entire deck is then a one-file change.
- Every chart script imports it. No inline colours, ever.
- `scale=2` on export for deck and print resolution.
- Subplots do not use `create_default_figure()` — they need per-panel axis
styling — so they import `FONT` and the colour constants and apply them in the
loop. Slightly more code, same output.
## Cursor specifics

Keep `README.md`, `data_map.md` and the current script open in tabs; they get
picked up as context automatically and the agent stops guessing at your
conventions.

Reference files explicitly with `@` when asking for changes — `@house_style.py`
beats describing it.

Ask for the verification script early rather than at the end. Having the agent
recompute its own quoted numbers is worth more than having it write more prose.

Point at `data_map.md` when asking anything about where data came from. It is
faster and more reliable than having the agent re-derive provenance from code.
## What to carry into the next project

Copy directly:

- `scripts/house_style.py` — unchanged
- `scripts/check_freshness.py` — edit the `CHARTS` dependency map, rest is generic
- `requirements.txt` — including the kaleido pin
- `.gitignore`
- The notebook bootstrap cell
- The folder skeleton above

Recreate per project:

- `README.md` with the pre-registered spec written before any run
- `data_map.md`, filled in as data lands rather than at the end
- `data/findings_log.csv` with the header, from run one

---

## This project's deviations (read these first)

- **venv.** From a clone, `python3 -m venv .venv` works normally and is what the README documents.
  On my own machine the working copy sits in a folder whose name contains `:`, the PATH separator,
  which `python -m venv` refuses to write into, so the venv lives at `~/.venvs/research` there.
  That is a local workaround, not a requirement.
- **`plotly==5.24.1` is pinned alongside `kaleido==0.2.1`.** Plotly 6.2+ / 7 refuse to export
  with kaleido 0.2.x and demand kaleido v1 -- the one that hangs. Keep the pair together.
- **Market data comes from `src/market_data.py`, not tvdatafeed.** See the README.
