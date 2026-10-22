# NFL Combine ↔ Career Lab

Streamlit app on top of your Postgres `NFL` database (players, combine_results,
combine_tracking, player_career_successes).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Connection: fill the sidebar form, or set `PGHOST/PGPORT/PGDBNAME/PGUSER/PGPASSWORD`,
or create `.streamlit/secrets.toml`:

```toml
[postgres]
host = "localhost"
port = "5432"
dbname = "NFL"
user = "postgres"
password = "..."
```

The app only issues read queries (the SQL console is restricted to a single
SELECT/WITH inside a read-only transaction).

## Tabs

- **Dashboard builder** – add/edit/reorder/duplicate panels (scatter, histogram, box/violin,
  aggregated bar, 2D binned heatmap, correlation heatmap, top-N table); presets; save/load layout as JSON.
- **Relationship finder** – scans every feature × outcome pair (Spearman/Pearson), with
  Benjamini–Hochberg FDR and an optional "draft pick partialled out" version.
- **Cohorts** – within-position quantiles of a metric vs an outcome.
- **Model & hidden gems** – cross-validated comparison of feature sets (pick only / combine /
  combine + tracking …), permutation importance, and players who over/under-perform the model.
- **Tracking viewer** – overlay runs (x/y path, speed or acceleration vs time) for any drill.
- **SQL console** – ad-hoc read-only queries with CSV download and quick chart.
- **Data** – the filtered master table and missingness.

## Derived columns

- Combine: `bmi`, `speed_score`, `burst` (vertical + broad jump), `agility_sum`, `age_at_draft`,
  and `*_pct` = percentile within position (higher = better, times are inverted).
- Tracking (40-yd dash, best attempt, `PLAYER` rows only): `trk40_vmax`, `trk40_t_vmax`, `trk40_amax`,
  `trk40_v1s/v2s`, `trk40_t10/20/30/40`, splits, `trk40_speed_hold`. Times are measured from the first
  frame with speed > 1 yd/s, so they differ from official gun-start times.
- Per-drill (sidebar): max speed, max acceleration, shortest duration, mean distance.
  `DRILL_ALIASES` in `nfl_data.py` merges drill names that look like renames (e.g. `LINE` → `LINE_DRILL`);
  edit it if they are actually different drills.
- Career: totals, per-season rates, `start_rate`, `all_pro_pts`, and `success_index`
  (mean percentile within position × draft year of the metrics you choose in the sidebar).

## Caveats

- Draft classes 2023–2025 only: later classes have 1–3 seasons, so career outcomes are noisy.
  Per-season metrics and the "last season" setting help, but treat results as exploratory.
- Undrafted players get pick = max pick + 1 for modelling (`pick_filled`, `log_pick`).
- Many pairs are tested at once; trust relationships that survive FDR correction and the draft-pick control.
