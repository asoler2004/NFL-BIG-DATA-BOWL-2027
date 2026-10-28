"""NFL Combine ↔ Career lab — Streamlit app.

Run:  streamlit run app2.py
"""
from __future__ import annotations

import json
import os
import uuid

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nfl_data as nd

st.set_page_config(page_title="NFL Combine ↔ Career Lab", page_icon="🏈", layout="wide")

# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #

def _secret(key: str, default: str) -> str:
    try:
        return str(st.secrets["postgres"].get(key, default))
    except Exception:
        return os.environ.get(f"PG{key.upper()}", default)


@st.cache_resource(show_spinner=False)
def get_engine(host, port, dbname, user, password):
    eng = nd.make_engine(host, port, dbname, user, password)
    with eng.connect() as c:
        c.exec_driver_sql("SELECT 1")
    return eng


with st.sidebar:
    st.title("🏈 NFL Lab")
    with st.expander("Database connection", expanded="engine_cfg" not in st.session_state):
        host = st.text_input("Host", _secret("host", "localhost"))
        port = st.text_input("Port", _secret("port", "5432"))
        dbname = st.text_input("Database", _secret("dbname", "NFL"))
        user = st.text_input("User", _secret("user", "postgres"))
        password = st.text_input("Password", _secret("password", ""), type="password")
        if st.button("Connect", type="primary", width="stretch"):
            st.session_state["engine_cfg"] = (host, port, dbname, user, password)
            st.cache_data.clear()

if "engine_cfg" not in st.session_state:
    # auto-connect when credentials come from secrets / env
    if _secret("password", "") or os.environ.get("PGHOST"):
        st.session_state["engine_cfg"] = (host, port, dbname, user, password)
    else:
        st.info("Enter your connection details in the sidebar and press **Connect**.")
        st.stop()

try:
    ENGINE = get_engine(*st.session_state["engine_cfg"])
except Exception as e:  # noqa: BLE001
    st.error(f"Could not connect: {e}")
    st.session_state.pop("engine_cfg", None)
    st.stop()

# --------------------------------------------------------------------------- #
# Data (cached)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading players, combine and career…", ttl=3600)
def cached_master(_engine, cfg):
    return nd.load_master(_engine)


@st.cache_data(show_spinner="Summarising tracking drills…", ttl=3600)
def cached_tracking_summary(_engine, cfg):
    return nd.load_tracking_summary(_engine)


@st.cache_data(show_spinner="Computing 40-yard-dash features from tracking…", ttl=3600)
def cached_forty(_engine, cfg):
    return nd.load_forty_features(_engine)


@st.cache_data(show_spinner="Summarising 2023–2025 game tracking…", ttl=3600)
def cached_game_tracking(_engine, cfg):
    return nd.load_game_tracking_summary(_engine)


@st.cache_data(show_spinner="Loading selected combine tracking frames…", ttl=3600)
def cached_combine_frames(_engine, cfg, drill_name, nfl_ids):
    return nd.load_frames(_engine, drill_name, list(nfl_ids))


@st.cache_data(show_spinner="Loading selected game tracking frames…", ttl=3600)
def cached_game_frames(_engine, cfg, nfl_ids):
    return nd.load_game_tracking_frames(_engine, list(nfl_ids))


cfg = st.session_state["engine_cfg"][:4]  # cache key without password
base = cached_master(ENGINE, cfg)
trk_summary = cached_tracking_summary(ENGINE, cfg)
forty = cached_forty(ENGINE, cfg)
game_tracking = cached_game_tracking(ENGINE, cfg)

# --------------------------------------------------------------------------- #
# Sidebar: career definition, tracking drills, filters
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.subheader("Career definition")
    last_season = st.number_input("Last season in data", 2023, 2030, 2025,
                                  help="Used to count seasons since the draft and normalise career totals per season.")
    idx_metrics = st.multiselect(
        "Success index = mean percentile (within position × draft year) of…",
        ["games_started_per_season", "snaps_per_season", "start_rate", "games_active_per_season",
         "snaps_per_game", "all_pro_pts", "pro_bowl_original_ballot"],
        default=["games_started_per_season", "snaps_per_season", "start_rate"],
    )

    st.subheader("Tracking drills")
    drill_counts = trk_summary.groupby("drill_name")["nfl_id"].nunique().sort_values(ascending=False)
    drills = st.multiselect(
        "Add per-drill features (speed, acceleration, angle, duration, distance)",
        [d for d in drill_counts.index if d != "FORTY_YARD_DASH"],
        format_func=lambda d: f"{d} ({drill_counts[d]})",
    )

master = nd.add_career_metrics(base, last_season=int(last_season))
master["success_index"] = nd.success_index(master, idx_metrics)
master = master.merge(forty, on="nfl_id", how="left")
master = master.merge(game_tracking, on="nfl_id", how="left")
drill_wide = nd.pivot_drills(trk_summary, drills)
if len(drill_wide.columns) > 1:
    master = master.merge(drill_wide, on="nfl_id", how="left")
DRILL_COLS = [c for c in master.columns if c.startswith("drill_")]

with st.sidebar:
    st.subheader("Filters")
    yrs = [str(y) for y in sorted(master["draft_year"].dropna().astype(int).unique())]
    f_years = [int(y) for y in st.multiselect("Draft year", yrs, default=yrs)]
    groups = sorted(master["pos_group"].unique())
    f_groups = st.multiselect("Position group", groups, default=groups)
    pos_opts = sorted(master.loc[master["pos_group"].isin(f_groups), "position"].unique())
    f_pos = st.multiselect("Position", pos_opts, default=pos_opts)
    f_status = st.radio("Draft status", ["All", "Drafted", "Undrafted"], horizontal=True)
    rounds = [str(r) for r in sorted(master["draft_round"].dropna().astype(int).unique())]
    f_rounds = [int(r) for r in st.multiselect("Draft round", rounds, default=rounds)]
    if master["draft_overall_pick"].notna().any():
        lo, hi = int(master["draft_overall_pick"].min()), int(master["draft_overall_pick"].max())
        f_pick = st.slider("Overall pick", lo, hi, (lo, hi))
    else:
        f_pick = None
    confs = sorted(master["college_conference"].unique())
    f_conf = st.multiselect("College conference", confs, default=confs)
    if master["weight"].notna().any():
        wlo, whi = int(master["weight"].min()), int(master["weight"].max())
        f_weight = st.slider("Weight (lb)", wlo, whi, (wlo, whi))
    else:
        f_weight = None
    f_min_games = st.slider("Min career games active", 0, int(master["career_games_active"].max() or 0), 0)
    f_need = st.multiselect("Require non-missing", [c for c in nd.COMBINE_RAW + nd.TRACK40 if c in master.columns])

df = master[
    master["draft_year"].isin(f_years) & master["pos_group"].isin(f_groups) & master["position"].isin(f_pos)
    & master["college_conference"].isin(f_conf)
].copy()
if f_status == "Drafted":
    df = df[df["undrafted"] == 0]
elif f_status == "Undrafted":
    df = df[df["undrafted"] == 1]
df = df[df["draft_round"].isin(f_rounds) | df["draft_round"].isna()]
if f_pick and f_status != "Undrafted":
    df = df[df["draft_overall_pick"].between(*f_pick) | df["draft_overall_pick"].isna()]
if f_weight:
    df = df[df["weight"].between(*f_weight) | df["weight"].isna()]
df = df[df["career_games_active"].fillna(0) >= f_min_games]
for c in f_need:
    df = df[df[c].notna()]

# --------------------------------------------------------------------------- #
# Column catalogue for pickers
# --------------------------------------------------------------------------- #

UTILITY_COLS = ["movement_translation_skill_index", "movement_translation_trench_index"]
NUMERIC = [c for c in (nd.COMBINE_RAW + nd.COMBINE_DERIVED + nd.COMBINE_PCT + nd.TRACK40 + nd.GAME_TRACK + DRILL_COLS
                       + nd.DRAFT_NUM + nd.CAREER_RAW + nd.CAREER_DERIVED + UTILITY_COLS) if c in df.columns]
CAREER_COLS = [c for c in nd.CAREER_RAW + nd.CAREER_DERIVED if c in df.columns]
FEATURE_COLS = [c for c in nd.COMBINE_RAW + nd.COMBINE_DERIVED + nd.COMBINE_PCT + nd.TRACK40 + nd.GAME_TRACK + DRILL_COLS + UTILITY_COLS
                if c in df.columns]
CATS = [c for c in nd.CATEGORICAL if c in df.columns]
HOVER = [c for c in ["display_name", "position", "draft_year", "draft_overall_pick"] if c in df.columns]
CORRELATION_GROUPS = {
    "Combine results": nd.COMBINE_RAW,
    "Derived ratios & indices": nd.COMBINE_DERIVED,
    "40-yard tracking": nd.TRACK40,
    "Selected drill tracking": DRILL_COLS,
    "NFL game tracking (2023–2025)": nd.GAME_TRACK,
    "NFL career outcomes": CAREER_COLS,
    "Draft context": ["draft_overall_pick", "draft_round", "draft_pick_within_round", "undrafted"],
}

st.caption(f"**{len(df):,}** of {len(master):,} players match the filters · "
           f"{df['career_games_active'].notna().sum():,} with career data · "
           f"{df['trk40_vmax'].notna().sum():,} with tracked 40-yd dash")


def pick(label, options, key, index=0, none=False, **kw):
    opts = ([None] if none else []) + list(options)
    i = min(index + (1 if none else 0), len(opts) - 1) if opts else 0
    return st.selectbox(label, opts, index=i, key=key, format_func=lambda x: "—" if x is None else x, **kw)


def default_idx(options, name):
    return options.index(name) if name in options else 0


# --------------------------------------------------------------------------- #
# Dashboard panels
# --------------------------------------------------------------------------- #

KINDS = {
    "scatter": "Scatter (X vs Y)",
    "histogram": "Histogram",
    "box": "Box / violin by group",
    "bar": "Aggregated bar (group → mean/median/count)",
    "heatmap2d": "2D binned heatmap (X × Y → Z)",
    "corr": "Correlation heatmap (features × outcomes)",
    "table": "Table (top N)",
}


def new_spec(kind: str) -> dict:
    s = {"id": uuid.uuid4().hex[:8], "kind": kind, "title": KINDS[kind].split(" (")[0], "full": False}
    f = lambda n, d: n if n in NUMERIC else d  # noqa: E731
    if kind == "scatter":
        s.update(x=f("forty", NUMERIC[0]), y=f("games_started_per_season", NUMERIC[-1]), color="pos_group",
                 trend="ols", size=None)
    elif kind == "histogram":
        s.update(x=f("success_index", NUMERIC[0]), color=None, bins=30, norm=False)
    elif kind == "box":
        s.update(x="pos_group", y=f("success_index", NUMERIC[0]), color=None, violin=False)
    elif kind == "bar":
        s.update(x="pos_group", y=f("success_index", NUMERIC[0]), agg="mean", color=None, bins=5)
    elif kind == "heatmap2d":
        s.update(x=f("forty", NUMERIC[0]), y=f("weight", NUMERIC[0]), z=f("success_index", NUMERIC[0]),
                 agg="mean", bins=8, min_n=3)
    elif kind == "corr":
        s.update(rows=[c for c in ["forty_pct", "vertical_pct", "broad_jump_pct", "three_cone_pct",
                                   "short_shuttle_pct", "bench_reps_pct", "weight_pct", "speed_score_pct"]
                       if c in NUMERIC],
                 cols=[c for c in ["games_started_per_season", "snaps_per_season", "start_rate", "success_index"]
                       if c in NUMERIC],
                 method="spearman", control_pick=False, min_n=15)
    elif kind == "table":
        s.update(sort=f("success_index", NUMERIC[0]), asc=False, n=20,
                 cols=[c for c in ["display_name", "position", "draft_overall_pick", "forty", "success_index"]
                       if c in df.columns])
    return s


def spec_editor(s: dict, k: str) -> dict:
    """Render widgets for a spec; returns the updated spec."""
    s = dict(s)
    s["title"] = st.text_input("Title", s["title"], key=f"{k}_title")
    kind = s["kind"]
    if kind == "scatter":
        c1, c2 = st.columns(2)
        s["x"] = c1.selectbox("X", NUMERIC, default_idx(NUMERIC, s["x"]), key=f"{k}_x")
        s["y"] = c2.selectbox("Y", NUMERIC, default_idx(NUMERIC, s["y"]), key=f"{k}_y")
        copts = [None] + CATS + NUMERIC
        s["color"] = st.selectbox("Color", copts, copts.index(s["color"]) if s["color"] in copts else 0,
                                  key=f"{k}_c", format_func=lambda x: "—" if x is None else x)
        sopts = [None] + NUMERIC
        s["size"] = st.selectbox("Size", sopts, sopts.index(s.get("size")) if s.get("size") in sopts else 0,
                                 key=f"{k}_s", format_func=lambda x: "—" if x is None else x)
        topts = [None, "ols", "lowess"]
        s["trend"] = st.selectbox("Trendline", topts, topts.index(s.get("trend")), key=f"{k}_t",
                                  format_func=lambda x: "none" if x is None else x)
    elif kind == "histogram":
        s["x"] = st.selectbox("Variable", NUMERIC, default_idx(NUMERIC, s["x"]), key=f"{k}_x")
        copts = [None] + CATS
        s["color"] = st.selectbox("Split by", copts, copts.index(s["color"]) if s["color"] in copts else 0,
                                  key=f"{k}_c", format_func=lambda x: "—" if x is None else x)
        s["bins"] = st.slider("Bins", 5, 80, int(s["bins"]), key=f"{k}_b")
        s["norm"] = st.checkbox("Normalise (density)", s["norm"], key=f"{k}_n")
    elif kind == "box":
        gopts = CATS + [c for c in NUMERIC if c in ("draft_round",)]
        s["x"] = st.selectbox("Group", gopts, default_idx(gopts, s["x"]), key=f"{k}_x")
        s["y"] = st.selectbox("Value", NUMERIC, default_idx(NUMERIC, s["y"]), key=f"{k}_y")
        copts = [None] + CATS
        s["color"] = st.selectbox("Color", copts, copts.index(s["color"]) if s["color"] in copts else 0,
                                  key=f"{k}_c", format_func=lambda x: "—" if x is None else x)
        s["violin"] = st.checkbox("Violin", s["violin"], key=f"{k}_v")
    elif kind == "bar":
        xopts = CATS + NUMERIC
        s["x"] = st.selectbox("Group by (numeric → quantile bins)", xopts, default_idx(xopts, s["x"]), key=f"{k}_x")
        s["y"] = st.selectbox("Value", NUMERIC, default_idx(NUMERIC, s["y"]), key=f"{k}_y")
        aopts = ["mean", "median", "count", "sum"]
        s["agg"] = st.selectbox("Aggregate", aopts, aopts.index(s["agg"]), key=f"{k}_a")
        copts = [None] + CATS
        s["color"] = st.selectbox("Split by", copts, copts.index(s["color"]) if s["color"] in copts else 0,
                                  key=f"{k}_c", format_func=lambda x: "—" if x is None else x)
        s["bins"] = st.slider("Quantile bins (numeric group)", 2, 10, int(s["bins"]), key=f"{k}_b")
    elif kind == "heatmap2d":
        c1, c2, c3 = st.columns(3)
        s["x"] = c1.selectbox("X", NUMERIC, default_idx(NUMERIC, s["x"]), key=f"{k}_x")
        s["y"] = c2.selectbox("Y", NUMERIC, default_idx(NUMERIC, s["y"]), key=f"{k}_y")
        s["z"] = c3.selectbox("Z (outcome)", NUMERIC, default_idx(NUMERIC, s["z"]), key=f"{k}_z")
        aopts = ["mean", "median", "count"]
        s["agg"] = st.selectbox("Aggregate", aopts, aopts.index(s["agg"]), key=f"{k}_a")
        s["bins"] = st.slider("Bins per axis", 3, 15, int(s["bins"]), key=f"{k}_b")
        s["min_n"] = st.slider("Hide cells with n <", 1, 15, int(s["min_n"]), key=f"{k}_mn")
    elif kind == "corr":
        s["rows"] = st.multiselect("Features (rows)", FEATURE_COLS, [c for c in s["rows"] if c in FEATURE_COLS],
                                   key=f"{k}_r")
        s["cols"] = st.multiselect("Outcomes (columns)", CAREER_COLS, [c for c in s["cols"] if c in CAREER_COLS],
                                   key=f"{k}_o")
        mopts = ["pearson", "spearman"]
        s["method"] = st.radio("Method", mopts, mopts.index(s["method"]), horizontal=True, key=f"{k}_m")
        s["control_pick"] = st.checkbox("Control for draft pick (log)", s["control_pick"], key=f"{k}_cp")
        s["min_n"] = st.slider("Min n per pair", 5, 60, int(s["min_n"]), key=f"{k}_mn")
    elif kind == "table":
        allc = list(df.columns)
        s["cols"] = st.multiselect("Columns", allc, [c for c in s["cols"] if c in allc], key=f"{k}_cols")
        s["sort"] = st.selectbox("Sort by", NUMERIC, default_idx(NUMERIC, s["sort"]), key=f"{k}_sort")
        s["asc"] = st.checkbox("Ascending", s["asc"], key=f"{k}_asc")
        s["n"] = st.slider("Rows", 5, 100, int(s["n"]), key=f"{k}_n")
    s["full"] = st.checkbox("Full width", s.get("full", False), key=f"{k}_full")
    return s


def _group_bins(frame: pd.DataFrame, col: str, bins: int) -> pd.Series:
    if col in nd.CATEGORICAL or frame[col].nunique() <= 8:
        return frame[col].astype(str)
    try:
        return pd.qcut(frame[col], bins, duplicates="drop").astype(str)
    except ValueError:
        return frame[col].astype(str)


def render_panel(frame: pd.DataFrame, s: dict):
    kind = s["kind"]
    if frame.empty:
        st.info("No data after filters.")
        return
    try:
        if kind == "scatter":
            cols = [s["x"], s["y"]] + [c for c in (s["color"], s["size"]) if c]
            d = frame.dropna(subset=[c for c in [s["x"], s["y"], s.get("size")] if c])
            fig = px.scatter(d, x=s["x"], y=s["y"], color=s["color"], size=s.get("size"),
                             trendline=s.get("trend"), hover_data=HOVER, opacity=0.75)
            r, p, n = nd.corr_pair(d[s["x"]], d[s["y"]], "spearman", 5)
            st.caption(f"n={n} · Spearman ρ={r:.2f} (p={p:.3g})" if n >= 5 and not np.isnan(r) else f"n={n}")
        elif kind == "histogram":
            fig = px.histogram(frame.dropna(subset=[s["x"]]), x=s["x"], color=s["color"], nbins=int(s["bins"]),
                               histnorm="probability density" if s["norm"] else None, barmode="overlay",
                               opacity=0.7)
        elif kind == "box":
            d = frame.dropna(subset=[s["y"]]).copy()
            d[s["x"]] = d[s["x"]].astype(str)
            fn = px.violin if s["violin"] else px.box
            fig = fn(d, x=s["x"], y=s["y"], color=s["color"], points="all", hover_data=HOVER)
        elif kind == "bar":
            d = frame.dropna(subset=[s["y"], s["x"]]).copy()
            d["_g"] = _group_bins(d, s["x"], int(s["bins"]))
            keys = ["_g"] + ([s["color"]] if s["color"] else [])
            g = d.groupby(keys)[s["y"]].agg(["mean", "median", "count", "sum", "sem"]).reset_index()
            if s["agg"] == "mean":
                err = g["sem"]
            else:
                err = None
            order = None
            if s["x"] in nd.CATEGORICAL or d[s["x"]].nunique() <= 8:
                order = sorted(g["_g"].unique())
            else:
                order = list(d.groupby("_g")[s["x"]].min().sort_values().index)
            fig = px.bar(g, x="_g", y=s["agg"], color=s["color"], barmode="group", error_y=err,
                         hover_data={"count": True}, category_orders={"_g": order})
            fig.update_layout(xaxis_title=s["x"], yaxis_title=f"{s['agg']} of {s['y']}")
        elif kind == "heatmap2d":
            d = frame.dropna(subset=[s["x"], s["y"], s["z"]]).copy()
            d["_x"] = pd.qcut(d[s["x"]], int(s["bins"]), duplicates="drop")
            d["_y"] = pd.qcut(d[s["y"]], int(s["bins"]), duplicates="drop")
            agg = d.groupby(["_y", "_x"], observed=True)[s["z"]].agg(["mean", "median", "count"]).reset_index()
            agg.loc[agg["count"] < int(s["min_n"]), ["mean", "median"]] = np.nan
            val = "count" if s["agg"] == "count" else s["agg"]
            mat = agg.pivot(index="_y", columns="_x", values=val)
            cnt = agg.pivot(index="_y", columns="_x", values="count")
            fig = go.Figure(go.Heatmap(
                z=mat.to_numpy(float), x=[str(c) for c in mat.columns], y=[str(i) for i in mat.index],
                customdata=cnt.to_numpy(), colorscale="RdYlGn", colorbar=dict(title=f"{val} {s['z']}"),
                hovertemplate=f"{s['x']}: %{{x}}<br>{s['y']}: %{{y}}<br>{val}: %{{z:.2f}}<br>n=%{{customdata}}<extra></extra>"))
            fig.update_layout(xaxis_title=s["x"], yaxis_title=s["y"])
        elif kind == "corr":
            if not s["rows"] or not s["cols"]:
                st.info("Choose at least one feature and one outcome.")
                return
            tbl = nd.corr_table(frame, s["rows"], s["cols"], s["method"], int(s["min_n"]),
                                ["log_pick"] if s["control_pick"] else None)
            if tbl.empty:
                st.info("Not enough data.")
                return
            r = nd.corr_matrix(tbl, "r").reindex(index=s["rows"], columns=s["cols"])
            q = nd.corr_matrix(tbl, "q").reindex(index=s["rows"], columns=s["cols"])
            n = nd.corr_matrix(tbl, "n").reindex(index=s["rows"], columns=s["cols"])
            txt = r.round(2).astype(str).where(q >= 0.05, r.round(2).astype(str) + "*")
            fig = go.Figure(go.Heatmap(
                z=r.to_numpy(float), x=list(r.columns), y=list(r.index), zmin=-1, zmax=1, colorscale="RdBu",
                reversescale=True, text=txt.to_numpy(), texttemplate="%{text}", customdata=n.to_numpy(),
                hovertemplate="%{y} → %{x}<br>r=%{z:.2f}<br>n=%{customdata}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            st.caption("* = significant after Benjamini–Hochberg FDR correction (q<0.05). "
                       + ("Draft pick partialled out." if s["control_pick"] else ""))
        elif kind == "table":
            d = frame.sort_values(s["sort"], ascending=s["asc"]).head(int(s["n"]))
            st.dataframe(d[s["cols"]] if s["cols"] else d, width="stretch", hide_index=True)
            return
        else:
            return
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420, legend_title_text="")
        st.plotly_chart(fig, width="stretch", key=f"fig_{s['id']}")
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not draw this panel: {e}")


PRESETS = {
    "Athleticism → outcomes": [
        ("corr", dict(title="Which combine metrics relate to career success?", full=True)),
        ("heatmap2d", dict(title="Speed × size → success index")),
        ("scatter", dict(title="40-yd (official) vs tracked top speed", x="forty", y="trk40_vmax")),
    ],
    "Movement translation": [
        ("scatter", dict(title="Skill movement index vs career success", x="movement_translation_skill_index", y="success_index")),
        ("box", dict(title="Skill movement index by position group", x="pos_group", y="movement_translation_skill_index")),
        ("corr", dict(title="Movement translation vs NFL outcomes", rows=["movement_translation_skill_index", "movement_translation_trench_index"], cols=["success_index", "games_started_per_season", "start_rate"], control_pick=True, method="spearman")),
    ],
    "Does the combine add anything beyond draft pick?": [
        ("corr", dict(title="Raw correlations", control_pick=False)),
        ("corr", dict(title="Draft pick partialled out", control_pick=True)),
        ("scatter", dict(title="Draft pick vs success", x="draft_overall_pick", y="success_index")),
    ],
    "Position profiles": [
        ("box", dict(title="Success index by position group")),
        ("bar", dict(title="Mean 40-yd percentile of starters", x="start_rate", y="forty_pct", bins=4)),
        ("table", dict(title="Top success index", full=True)),
    ],
}


def load_preset(name: str):
    out = []
    for kind, over in PRESETS[name]:
        s = new_spec(kind)
        s.update({k: v for k, v in over.items() if (k not in ("x", "y", "z") or v in NUMERIC)})
        out.append(s)
    st.session_state["panels"] = out


st.session_state.setdefault("panels", [])

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

tab_dash, tab_rel, tab_corr, tab_cohort, tab_model, tab_profile, tab_trk, tab_insights, tab_sql, tab_data = st.tabs(
    ["📊 Dashboard builder", "🔗 Relationship finder", "🧩 Correlation matrix", "🎯 Cohorts",
     "🤖 Model & hidden gems", "👤 Player profile", "🏃 Tracking viewer", "📈 Jerk & spectrum",
     "🧮 SQL console", "🗂 Data"]
)

# ---- Dashboard builder ---------------------------------------------------- #
with tab_dash:
    top = st.columns([2, 2, 1, 2])
    pre = top[0].selectbox("Start from a preset", ["—"] + list(PRESETS), key="preset_sel")
    if top[1].button("Load preset", disabled=pre == "—"):
        load_preset(pre)
        st.rerun()
    ncols = top[2].selectbox("Columns", [1, 2, 3], index=1)
    if top[3].button("Clear dashboard"):
        st.session_state["panels"] = []
        st.rerun()

    with st.expander("➕ Add panel", expanded=not st.session_state["panels"]):
        kind = st.selectbox("Panel type", list(KINDS), format_func=lambda k: KINDS[k], key="add_kind")
        draft_key = f"draft_{kind}"
        if draft_key not in st.session_state:
            st.session_state[draft_key] = new_spec(kind)
        draft = spec_editor(st.session_state[draft_key], f"add_{kind}")
        st.session_state[draft_key] = draft
        with st.container(border=True):
            render_panel(df, draft)
        if st.button("Add to dashboard", type="primary"):
            st.session_state["panels"].append(dict(draft, id=uuid.uuid4().hex[:8]))
            st.session_state.pop(draft_key, None)
            st.rerun()

    panels = st.session_state["panels"]
    if panels:
        rows, buf = [], []
        for i, p in enumerate(panels):
            if p.get("full"):
                if buf:
                    rows.append(buf)
                    buf = []
                rows.append([i])
            else:
                buf.append(i)
                if len(buf) == ncols:
                    rows.append(buf)
                    buf = []
        if buf:
            rows.append(buf)

        for row in rows:
            cols = st.columns(len(row))
            for col, i in zip(cols, row):
                p = panels[i]
                with col, st.container(border=True):
                    h = st.columns([6, 1, 1, 1, 1])
                    h[0].markdown(f"**{p['title']}**")
                    if h[1].button("⬆", key=f"up_{p['id']}", disabled=i == 0):
                        panels[i - 1], panels[i] = panels[i], panels[i - 1]
                        st.rerun()
                    if h[2].button("⬇", key=f"dn_{p['id']}", disabled=i == len(panels) - 1):
                        panels[i + 1], panels[i] = panels[i], panels[i + 1]
                        st.rerun()
                    if h[3].button("⧉", key=f"cp_{p['id']}", help="Duplicate"):
                        panels.insert(i + 1, dict(p, id=uuid.uuid4().hex[:8], title=p["title"] + " (copy)"))
                        st.rerun()
                    if h[4].button("✕", key=f"rm_{p['id']}", help="Remove"):
                        panels.pop(i)
                        st.rerun()
                    render_panel(df, p)
                    with st.expander("Edit"):
                        panels[i] = spec_editor(p, f"ed_{p['id']}")

    with st.expander("💾 Save / load dashboard layout"):
        st.download_button("Download layout (JSON)", json.dumps(st.session_state["panels"], indent=2),
                           "nfl_dashboard.json", "application/json")
        up = st.file_uploader("Load layout", type="json")
        if up is not None and st.button("Apply uploaded layout"):
            try:
                loaded = json.load(up)
                for s in loaded:
                    s["id"] = uuid.uuid4().hex[:8]
                st.session_state["panels"] = loaded
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Invalid layout: {e}")

# ---- Relationship finder -------------------------------------------------- #
with tab_corr:
    st.markdown(
        "Explore pairwise correlations across combine tests, body ratios, combine tracking summaries, "
        "2023–2025 NFL game tracking, and career outcomes. Game tracking includes per-player peak/minimum "
        "speed and acceleration, equal-play-weighted means, max direction change, and per-play distance."
    )
    controls = st.columns(4)
    corr_method = controls[0].radio("Correlation", ["spearman", "pearson"], horizontal=True, key="matrix_method")
    corr_min_n = controls[1].slider("Minimum players per pair", 5, 100, 15, key="matrix_min_n")
    corr_scope = controls[2].selectbox(
        "Position scope", ["All filtered players"] + sorted(df["pos_group"].dropna().unique()),
        key="matrix_scope",
    )
    corr_control_pick = controls[3].checkbox("Control for log draft pick", key="matrix_control_pick")

    groups_selected = st.multiselect(
        "Metric groups",
        list(CORRELATION_GROUPS),
        default=[name for name in CORRELATION_GROUPS if name != "Draft context"],
        key="matrix_groups",
    )
    available_matrix_cols = list(dict.fromkeys(
        col for group in groups_selected for col in CORRELATION_GROUPS[group] if col in df.columns
    ))
    matrix_cols = st.multiselect(
        "Metrics in matrix",
        available_matrix_cols,
        default=available_matrix_cols,
        key="matrix_metrics",
        help="Remove individual metrics to make a smaller, easier-to-read matrix.",
    )
    annotate_matrix = st.checkbox("Annotate correlation values", value=len(matrix_cols) <= 30, key="matrix_annotate")

    if corr_control_pick and "log_pick" in matrix_cols:
        st.warning("Remove log_pick from the matrix to control the other metrics for draft pick.")
        corr_control_pick = False
    if not matrix_cols:
        st.info("Select at least one metric group and one metric.")
    else:
        corr_df = df if corr_scope == "All filtered players" else df[df["pos_group"] == corr_scope]
        if corr_control_pick:
            corr_data = nd.residualize(corr_df, matrix_cols, ["log_pick"])
            st.caption("Displayed correlations use residuals after linearly controlling for log draft pick.")
        else:
            corr_data = corr_df[matrix_cols].apply(pd.to_numeric, errors="coerce")

        corr_values = pd.DataFrame(np.nan, index=matrix_cols, columns=matrix_cols, dtype=float)
        pair_counts = pd.DataFrame(0, index=matrix_cols, columns=matrix_cols, dtype=int)
        for i, left in enumerate(matrix_cols):
            for right in matrix_cols[i:]:
                r, _, n = nd.corr_pair(corr_data[left], corr_data[right], corr_method, corr_min_n)
                corr_values.loc[left, right] = corr_values.loc[right, left] = r
                pair_counts.loc[left, right] = pair_counts.loc[right, left] = n

        labels = corr_values.apply(
            lambda column: column.map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
        )
        combine_metric_cols = set(nd.COMBINE_RAW + nd.COMBINE_DERIVED + nd.COMBINE_PCT + nd.TRACK40 + DRILL_COLS)
        game_tracking_cols = set(nd.GAME_TRACK)
        professional_metric_cols = set(CAREER_COLS)
        category_colors = {
            "Combine": "#2878B5",
            "Game tracking": "#2A9D8F",
            "NFL career": "#E68632",
            "Draft context": "#8C8C8C",
        }
        metric_categories = {
            col: (
                "Combine" if col in combine_metric_cols else
                "Game tracking" if col in game_tracking_cols else
                "NFL career" if col in professional_metric_cols else
                "Draft context"
            )
            for col in matrix_cols
        }
        heatmap = go.Figure(go.Heatmap(
            z=corr_values.to_numpy(),
            x=matrix_cols,
            y=matrix_cols,
            customdata=pair_counts.to_numpy(),
            text=labels,
            texttemplate="%{text}" if annotate_matrix else None,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            zmid=0,
            colorbar=dict(title="r"),
            hovertemplate="%{y} × %{x}<br>r=%{z:.3f}<br>n=%{customdata}<extra></extra>",
        ))
        annotations = [
            dict(
                x=col, y=-0.04, xref="x", yref="paper",
                text=col, showarrow=False, textangle=-45,
                xanchor="right", yanchor="top",
                font=dict(color=category_colors[metric_categories[col]], size=11),
            )
            for col in matrix_cols
        ] + [
            dict(
                x=0.30, y=row, xref="paper", yref="y",
                text=row, showarrow=False,
                xanchor="right", yanchor="middle",
                font=dict(color=category_colors[metric_categories[row]], size=11),
            )
            for row in matrix_cols
        ]
        heatmap.update_layout(
            height=max(650, min(1600, len(matrix_cols) * 20)),
            margin=dict(l=10, r=10, t=20, b=180),
            annotations=annotations,
            xaxis=dict(
                domain=[0.32, 1],
                categoryorder="array",
                categoryarray=matrix_cols,
                showticklabels=False,
            ),
            yaxis=dict(
                domain=[0.12, 1],
                categoryorder="array",
                categoryarray=matrix_cols,
                showticklabels=False,
                autorange="reversed",
            ),
        )
        st.plotly_chart(heatmap, width="stretch")
        st.markdown(
            f"<span style='color:{category_colors['Combine']}'>■ Combine</span> &nbsp; "
            f"<span style='color:{category_colors['Game tracking']}'>■ NFL game tracking (2023–2025)</span> &nbsp; "
            f"<span style='color:{category_colors['NFL career']}'>■ NFL career</span> &nbsp; "
            f"<span style='color:{category_colors['Draft context']}'>■ Draft context</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{len(matrix_cols)} metrics · {len(corr_df):,} players in scope. "
            "Cells show pairwise-complete correlations; hover for each pair's sample size. "
            "Treat this exploratory matrix as descriptive, not causal."
        )
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "Download correlation matrix (CSV)",
            corr_values.to_csv(index=True),
            "nfl_correlation_matrix.csv",
            "text/csv",
        )
        dl2.download_button(
            "Download pair counts (CSV)",
            pair_counts.to_csv(index=True),
            "nfl_correlation_pair_counts.csv",
            "text/csv",
        )

# ---- Relationship finder -------------------------------------------------- #
with tab_rel:
    st.markdown("Scan **every** feature × outcome pair, rank them, and check whether they survive "
                "multiple-testing correction and controlling for draft pick.")
    c = st.columns(4)
    r_feats = c[0].multiselect("Feature groups", ["Combine raw", "Combine derived", "Combine percentile (within position)",
                                                  "Tracking 40-yd", "Tracking drills", "NFL game tracking"],
                               default=["Combine percentile (within position)", "Tracking 40-yd"])
    r_out = c[1].multiselect("Outcomes", CAREER_COLS,
                             default=[x for x in ["success_index", "games_started_per_season", "snaps_per_season",
                                                  "start_rate", "all_pro_pts"] if x in CAREER_COLS])
    r_method = c[2].radio("Method", ["spearman", "pearson"], horizontal=True)
    r_min = c[3].slider("Min n", 10, 100, 25)
    r_scope = st.multiselect("Run separately for position groups", sorted(df["pos_group"].unique()),
                             default=[], help="Empty = all filtered players together.")
    r_ctrl = st.checkbox("Also compute with draft pick partialled out", True)

    gmap = {"Combine raw": nd.COMBINE_RAW, "Combine derived": nd.COMBINE_DERIVED,
            "Combine percentile (within position)": nd.COMBINE_PCT, "Tracking 40-yd": nd.TRACK40,
            "Tracking drills": DRILL_COLS, "NFL game tracking": nd.GAME_TRACK}
    feats = [f for g in r_feats for f in gmap[g] if f in df.columns]
    if st.button("Scan relationships", type="primary") and feats and r_out:
        results = []
        scopes = r_scope or ["ALL"]
        for sc in scopes:
            sub = df if sc == "ALL" else df[df["pos_group"] == sc]
            base_t = nd.corr_table(sub, feats, r_out, r_method, r_min)
            base_t["scope"], base_t["control"] = sc, "none"
            results.append(base_t)
            if r_ctrl:
                ct = nd.corr_table(sub, feats, r_out, r_method, r_min, ["log_pick"])
                ct["scope"], ct["control"] = sc, "draft pick"
                results.append(ct)
        st.session_state["scan"] = pd.concat(results, ignore_index=True)
    scan = st.session_state.get("scan")
    if scan is not None and not scan.empty:
        sig = st.checkbox("Only FDR-significant (q<0.05)", False)
        view = scan.dropna(subset=["r"]).copy()
        if sig:
            view = view[view["q"] < 0.05]
        view["|r|"] = view["r"].abs()
        view = view.sort_values("|r|", ascending=False)
        st.dataframe(view[["scope", "control", "feature", "outcome", "r", "n", "p", "q"]].head(300),
                     width="stretch", hide_index=True,
                     column_config={"r": st.column_config.NumberColumn(format="%.3f"),
                                    "p": st.column_config.NumberColumn(format="%.4f"),
                                    "q": st.column_config.NumberColumn(format="%.4f")})
        top = view.head(1)
        if len(top):
            t = top.iloc[0]
            st.caption(f"Strongest pair: **{t['feature']} → {t['outcome']}** (r={t['r']:.2f}, n={int(t['n'])}, "
                       f"scope {t['scope']}, control: {t['control']}). Click a pair below to plot it.")
        pairs = view.head(50).apply(lambda r: f"{r['feature']} → {r['outcome']}", axis=1).tolist()
        if pairs:
            sel = st.selectbox("Plot pair", pairs)
            fx, fy = sel.split(" → ")
            fig = px.scatter(df.dropna(subset=[fx, fy]), x=fx, y=fy, color="pos_group", trendline="ols",
                             hover_data=HOVER, opacity=0.75)
            st.plotly_chart(fig, width="stretch")
        st.warning("With dozens of tests some pairs look strong by chance — prefer pairs that stay significant "
                   "after FDR correction **and** after controlling for draft pick, and that replicate across draft years.")

# ---- Cohorts -------------------------------------------------------------- #
with tab_cohort:
    st.markdown("Split players into **within-position** quantiles of a combine/tracking metric and compare outcomes.")
    c = st.columns(4)
    metric = c[0].selectbox("Metric", FEATURE_COLS, default_idx(FEATURE_COLS, "forty"))
    outcome = c[1].selectbox("Outcome", CAREER_COLS, default_idx(CAREER_COLS, "success_index"))
    q = c[2].slider("Quantiles", 2, 5, 3)
    by = c[3].selectbox("Compare within", ["position", "pos_group", "all players"], index=1)
    d = df.dropna(subset=[metric]).copy()
    if by == "all players":
        d["_q"] = pd.qcut(d[metric], q, labels=False, duplicates="drop")
    else:
        d["_q"] = d.groupby(by)[metric].transform(lambda x: pd.qcut(x.rank(method="first"), q, labels=False)
                                                  if x.notna().sum() >= q * 3 else pd.Series(np.nan, index=x.index))
    d = d.dropna(subset=["_q", outcome])
    if len(d):
        lower_better = metric in nd.LOWER_IS_BETTER or metric.endswith(("_t10", "_t20", "_t30", "_t40"))
        labels = {i: f"Q{int(i) + 1}" for i in sorted(d["_q"].unique())}
        d["quantile"] = d["_q"].map(labels)
        st.caption(("Q1 = lowest values" + (" (= fastest / best)" if lower_better else "")) + f" · n={len(d)}")
        g = d.groupby(["quantile"] + ([] if by == "all players" else ["pos_group"]))[outcome].agg(
            ["mean", "count", "sem"]).reset_index()
        fig = px.bar(g, x="quantile", y="mean", color=None if by == "all players" else "pos_group", barmode="group",
                     error_y="sem", hover_data=["count"])
        fig.update_layout(yaxis_title=f"mean {outcome}")
        st.plotly_chart(fig, width="stretch")
        top_q, bot_q = d[d["_q"] == d["_q"].max()][outcome], d[d["_q"] == d["_q"].min()][outcome]
        if len(top_q) > 3 and len(bot_q) > 3:
            from scipy import stats as sps
            u = sps.mannwhitneyu(top_q, bot_q, alternative="two-sided")
            st.caption(f"Top vs bottom quantile: median {top_q.median():.2f} vs {bot_q.median():.2f} "
                       f"(Mann-Whitney p={u.pvalue:.3g})")
    else:
        st.info("Not enough data for this combination.")

# ---- Model ---------------------------------------------------------------- #
with tab_model:
    st.markdown("Does athletic testing predict career outcomes **beyond draft position**? "
                "Cross-validated models compare feature sets; residuals highlight over-/under-performers.")
    c = st.columns(4)
    target = c[0].selectbox("Target", CAREER_COLS, default_idx(CAREER_COLS, "success_index"), key="m_t")
    kind = c[1].radio("Model", ["rf", "gb"], format_func={"rf": "Random forest", "gb": "Gradient boosting"}.get,
                      horizontal=True)
    min_cov = c[2].slider("Min feature coverage", 0.3, 1.0, 0.6, help="Drop features missing in more players than this.")
    add_pos = c[3].checkbox("Add position-group dummies", True)
    sets = {
        "Draft pick only": ["log_pick"],
        "Combine only (percentiles)": [x for x in nd.COMBINE_PCT if x in df.columns],
        "Combine + tracking 40": [x for x in nd.COMBINE_PCT + nd.TRACK40 if x in df.columns],
        "Pick + combine": ["log_pick"] + [x for x in nd.COMBINE_PCT if x in df.columns],
        "Pick + combine + tracking 40": ["log_pick"] + [x for x in nd.COMBINE_PCT + nd.TRACK40 if x in df.columns],
    }
    if DRILL_COLS:
        sets["Pick + combine + tracking 40 + drills"] = sets["Pick + combine + tracking 40"] + DRILL_COLS
    chosen = st.multiselect("Feature sets to compare", list(sets), default=list(sets)[:5])
    if st.button("Run models", type="primary") and chosen:
        d = df.dropna(subset=[target])
        if len(d) < 40:
            st.warning("Need at least 40 players with this target (loosen the filters).")
        else:
            res = {}
            prog = st.progress(0.0)
            for i, name in enumerate(chosen):
                cols = [c_ for c_ in sets[name] if d[c_].notna().mean() >= min_cov]
                X = d[cols].copy()
                if add_pos:
                    X = pd.concat([X, nd.one_hot_position(d)], axis=1)
                res[name] = nd.cv_model(X, d[target], kind=kind)
                res[name]["cols"] = cols
                prog.progress((i + 1) / len(chosen))
            st.session_state["model_res"] = (res, target, d[["nfl_id", "display_name", "position", "draft_year",
                                                             "draft_overall_pick", target]])
    if "model_res" in st.session_state:
        res, tgt, meta = st.session_state["model_res"]
        comp = pd.DataFrame({"feature set": list(res), "CV R²": [r["r2_mean"] for r in res.values()],
                             "± sd": [r["r2_std"] for r in res.values()], "n players": [r["n"] for r in res.values()]})
        fig = px.bar(comp, x="feature set", y="CV R²", error_y="± sd", text=comp["CV R²"].round(3))
        fig.update_layout(title=f"Predicting {tgt} (5-fold CV)")
        st.plotly_chart(fig, width="stretch")
        st.caption("If 'Pick + combine' barely beats 'Draft pick only', the combine adds little once teams have "
                   "priced it into the draft. Negative R² = worse than predicting the mean.")
        which = st.selectbox("Inspect model", list(res), index=len(res) - 1)
        c1, c2 = st.columns(2)
        imp = res[which]["importance"].head(15)[::-1]
        c1.plotly_chart(px.bar(imp, orientation="h", title="Permutation importance (ΔR² when shuffled)")
                        .update_layout(showlegend=False, xaxis_title="", yaxis_title=""), width="stretch")
        oof = res[which]["oof"].rename("predicted")
        out = meta.join(oof, how="inner")
        out["residual"] = out[tgt] - out["predicted"]
        out = out.rename(columns={tgt: "actual"})
        with c2:
            st.markdown("**Over-performers** (actual ≫ predicted)")
            st.dataframe(out.sort_values("residual", ascending=False).head(12)
                         [["display_name", "position", "draft_overall_pick", "actual", "predicted", "residual"]]
                         .round(2), hide_index=True, width="stretch")
            st.markdown("**Under-performers**")
            st.dataframe(out.sort_values("residual").head(12)
                         [["display_name", "position", "draft_overall_pick", "actual", "predicted", "residual"]]
                         .round(2), hide_index=True, width="stretch")

# ---- Player profile ------------------------------------------------------- #
with tab_profile:
    st.markdown("Explore one player's full athletic profile, career outcomes, and drill data.")
    if df.empty:
        st.info("No players match the current filters.")
    else:
        player_labels = {
            int(r.nfl_id): f"{r.display_name} ({r.position}, {int(r.draft_year)})"
            for r in df[["nfl_id", "display_name", "position", "draft_year"]].dropna(subset=["nfl_id"]).itertuples()
        }
        selected_id = st.selectbox("Player", list(player_labels), format_func=lambda i: player_labels.get(i, str(i)))
        player = df[df["nfl_id"] == selected_id].iloc[0]

        def fmt_profile_value(v):
            if pd.isna(v):
                return "—"
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                x = float(v)
                if np.isclose(x, round(x)):
                    return f"{x:.0f}"
                return f"{x:.2f}"
            return str(v)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"### {player['display_name']}")
        c2.metric("Position", player.get("position", "—"))
        c3.metric("Draft year", fmt_profile_value(player.get("draft_year")))
        c4.metric("Pick", fmt_profile_value(player.get("draft_overall_pick")))

        key_stats = {
            "forty": "40-yard",
            "vertical": "Vertical",
            "broad_jump": "Broad jump",
            "bench_reps": "Bench",
            "success_index": "Success index",
            "career_games_started": "Games started",
            "career_games_active": "Games active",
            "total_snaps": "Total snaps",
            "trk40_vmax": "Tracked max speed",
        }
        cards = st.columns(len(key_stats))
        for col, (metric, label) in zip(cards, key_stats.items()):
            with col:
                if metric in player.index:
                    st.metric(label, fmt_profile_value(player.get(metric)))
                else:
                    st.metric(label, "—")

        combine_cols = [c for c in nd.COMBINE_RAW + nd.COMBINE_DERIVED + nd.COMBINE_PCT + nd.TRACK40 if c in player.index]
        career_cols = [c for c in nd.CAREER_RAW + nd.CAREER_DERIVED if c in player.index]

        combine_df = pd.DataFrame({"metric": combine_cols, "value": [player[c] for c in combine_cols]}).dropna(subset=["value"])
        career_df = pd.DataFrame({"metric": career_cols, "value": [player[c] for c in career_cols]}).dropna(subset=["value"])

        st.subheader("Combine stats")
        st.dataframe(combine_df, hide_index=True, use_container_width=True)
        st.subheader("Career stats")
        st.dataframe(career_df, hide_index=True, use_container_width=True)

        available_metrics = [c for c in FEATURE_COLS if c in player.index and not pd.isna(player[c])]
        if not available_metrics:
            st.info(f"{player['display_name']} has no available feature values in the current filtered set.")
            available_metrics = ["forty"] if "forty" in df.columns else []
        metric_choice = st.selectbox("Combine metric to compare against peers",
                                    available_metrics,
                                    index=default_idx(available_metrics, "forty") if "forty" in available_metrics else 0)
        if pd.isna(player.get(metric_choice)):
            st.info(f"{player['display_name']} is missing a value for {metric_choice}.")
        else:
            peer = df.dropna(subset=[metric_choice]).copy()
            fig = px.histogram(peer, x=metric_choice, nbins=30, marginal="rug",
                               opacity=0.7, title=f"{metric_choice} distribution for current filtered cohort")
            fig.add_vline(x=float(player[metric_choice]), line_color="red", line_dash="dash",
                          annotation_text=f"{player['display_name']}", annotation_position="top left")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        player_drills = set(trk_summary.loc[trk_summary["nfl_id"] == selected_id, "drill_name"].dropna().tolist())
        drill_choices = [d for d in list(drill_counts.index) if d in player_drills and d != "FORTY_YARD_DASH"]
        if not drill_choices and "FORTY_YARD_DASH" in player_drills:
            drill_choices = ["FORTY_YARD_DASH"]
        if not drill_choices:
            st.info(f"{player['display_name']} has no tracking drill data in the current filters.")
        else:
            drill_choice = st.selectbox("Tracking drill", drill_choices,
                                       index=0 if "FORTY_YARD_DASH" not in drill_choices else drill_choices.index("FORTY_YARD_DASH"),
                                       format_func=lambda d: f"{d} ({drill_counts.get(d, 0)} players)")
            tracking_view = st.radio("Tracking plot", ["path", "speed", "acceleration"], horizontal=True)
            frames = nd.load_frames(ENGINE, drill_choice, [int(selected_id)])
            if frames.empty:
                st.info("No tracking frames are available for this player/drill.")
            else:
                if tracking_view == "path":
                    fig = px.line(frames, x="x", y="y", color="attempt", title=f"{drill_choice} path on the field")
                    fig.update_yaxes(scaleanchor="x")
                elif tracking_view == "speed":
                    fig = px.line(frames, x="t", y="s", color="attempt", title=f"{drill_choice} speed vs time")
                else:
                    fig = px.line(frames, x="t", y="a", color="attempt", title=f"{drill_choice} acceleration vs time")
                st.plotly_chart(fig, use_container_width=True)

        game_frames = nd.load_game_tracking_frames(ENGINE, [int(selected_id)])
        if game_frames.empty:
            st.info(f"{player['display_name']} has no in-game tracking data in the database.")
        else:
            play_choices = sorted(game_frames["play_label"].dropna().unique().tolist())
            play_limit = min(10, len(play_choices))
            selected_plays = st.multiselect(
                "Game tracking plays",
                play_choices,
                default=play_choices[:play_limit],
                help="Show the selected player's tracked game plays; the default focuses on the first 10 plays.",
            )
            if selected_plays:
                game_view = st.radio("Game tracking view", ["path", "speed", "acceleration"], horizontal=True, key="game_track_view")
                filtered = game_frames[game_frames["play_label"].isin(selected_plays)].copy()
                if game_view == "path":
                    fig = px.line(filtered, x="x", y="y", color="play_label", title=f"{player['display_name']} game path by play")
                    fig.update_yaxes(scaleanchor="x")
                elif game_view == "speed":
                    fig = px.line(filtered, x="t", y="s", color="play_label", title=f"{player['display_name']} game speed vs time")
                else:
                    fig = px.line(filtered, x="t", y="a", color="play_label", title=f"{player['display_name']} game acceleration vs time")
                st.plotly_chart(fig, use_container_width=True)

                play_summary = (
                    filtered.groupby("play_label", as_index=False)
                    .agg(
                        game_id=("game_id", "first"),
                        play_id=("play_id", "first"),
                        max_speed=("s", "max"),
                        max_accel=("a", "max"),
                        total_distance=("dis", "sum"),
                        duration_s=("t", "max"),
                    )
                    .sort_values("play_id")
                    .rename(columns={"play_label": "play"})
                )
                play_summary["game_id"] = play_summary["game_id"].astype(int)
                play_summary["play_id"] = play_summary["play_id"].astype(int)
                play_summary["max_speed"] = play_summary["max_speed"].round(2)
                play_summary["max_accel"] = play_summary["max_accel"].round(2)
                play_summary["total_distance"] = play_summary["total_distance"].round(2)
                play_summary["duration_s"] = play_summary["duration_s"].round(2)
                st.subheader("Selected game plays")
                st.dataframe(play_summary[["play", "game_id", "play_id", "max_speed", "max_accel", "total_distance", "duration_s"]], hide_index=True, use_container_width=True)
            else:
                st.info("Select at least one play to display the in-game tracking chart.")

# ---- Tracking viewer ------------------------------------------------------ #
with tab_trk:
    all_drills = list(drill_counts.index)
    dr = st.selectbox("Drill", all_drills, index=all_drills.index("FORTY_YARD_DASH") if "FORTY_YARD_DASH" in all_drills else 0,
                      format_func=lambda d_: f"{d_} ({drill_counts[d_]} players)")
    have = trk_summary.loc[trk_summary["drill_name"] == dr, "nfl_id"]
    cand = df[df["nfl_id"].isin(have)].sort_values("display_name")
    labels = {str(int(r.nfl_id)): f"{r.display_name} ({r.position}, {int(r.draft_year)})" for r in cand.itertuples()}
    # drop selections that have no run in the newly chosen drill (stale keys break the widget)
    st.session_state["trk_players"] = [p for p in st.session_state.get("trk_players", []) if p in labels][:6]
    chosen_ids = [int(i) for i in st.multiselect("Players (max 6)", list(labels),
                                                 format_func=lambda i: labels.get(i, str(i)),
                                                 max_selections=6, key="trk_players")]
    if chosen_ids:
        frames = nd.load_frames(ENGINE, dr, chosen_ids)
        frames["player"] = frames["nfl_id"].astype(int).astype(str).map(labels)
        frames["run"] = frames["player"] + " · attempt " + frames["attempt"].astype(str)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(frames, x="x", y="y", color="run", title="Path on the field (x, y)")
                        .update_yaxes(scaleanchor="x"), width="stretch")
        metric_t = st.radio("Time series", ["s", "a"], format_func={"s": "Speed", "a": "Acceleration"}.get, horizontal=True)
        c2.plotly_chart(px.line(frames, x="t", y=metric_t, color="run", title=f"{metric_t} vs time (s)"),
                        width="stretch")
        cols_show = [c_ for c_ in ["display_name", "position", "forty", "trk40_vmax", "trk40_t40", "success_index",
                                   "career_games_started", "career_games_active"] if c_ in df.columns]
        st.dataframe(df[df["nfl_id"].isin(chosen_ids)][cols_show], hide_index=True, width="stretch")
    else:
        st.info("Pick one or more players to overlay their runs.")

# ---- Jerk and spectrum insights ------------------------------------------ #
with tab_insights:
    st.markdown(
        "Inspect acceleration's time derivative (jerk) for combine drills or NFL game plays. "
        "Jerk is estimated from adjacent acceleration samples and is reported in yd/s³. "
        "Periodograms use linearly detrended tracks resampled at the median frame interval; "
        "frequency summaries are exploratory, not direct measures of player quality."
    )
    source_col, position_col = st.columns(2)
    tracking_source = source_col.radio(
        "Tracking source", ["Combine", "NFL game"], horizontal=True, key="insight_source"
    )
    insight_positions = position_col.multiselect(
        "Position groups",
        sorted(df["pos_group"].dropna().unique()),
        default=sorted(df["pos_group"].dropna().unique()),
        key="insight_positions",
    )
    insight_candidates = df[df["pos_group"].isin(insight_positions)].copy()

    if tracking_source == "Combine":
        insight_drills = sorted(trk_summary["drill_name"].dropna().unique())
        insight_drill = st.selectbox(
            "Combine drill",
            insight_drills,
            index=insight_drills.index("FORTY_YARD_DASH") if "FORTY_YARD_DASH" in insight_drills else 0,
            key="insight_drill",
        )
        tracked_ids = trk_summary.loc[trk_summary["drill_name"] == insight_drill, "nfl_id"]
        insight_candidates = insight_candidates[insight_candidates["nfl_id"].isin(tracked_ids)]
    else:
        insight_candidates = insight_candidates[
            insight_candidates["game_tracking_plays"].fillna(0).gt(0)
        ]

    insight_candidates = insight_candidates.sort_values("display_name")
    insight_labels = {
        int(row.nfl_id): f"{row.display_name} ({row.position}, {int(row.draft_year)})"
        for row in insight_candidates.itertuples()
    }
    st.session_state["insight_player_ids"] = [
        player_id for player_id in st.session_state.get("insight_player_ids", [])
        if player_id in insight_labels
    ][:6]
    chosen_insight_ids = st.multiselect(
        "Players (max 6)",
        list(insight_labels),
        format_func=lambda player_id: insight_labels[player_id],
        max_selections=6,
        key="insight_player_ids",
        help="Choose several players to compare runs and explore spectral features against their career metrics.",
    )
    if not chosen_insight_ids:
        st.info("Choose players from the selected position group(s) to load their tracking.")
    else:
        ids = tuple(int(player_id) for player_id in chosen_insight_ids)
        if tracking_source == "Combine":
            frames = cached_combine_frames(ENGINE, cfg, insight_drill, ids)
            run_cols = ["nfl_id", "drill_name", "attempt"]
            frames["track_label"] = (
                frames["nfl_id"].map(insight_labels)
                + " · attempt "
                + frames["attempt"].astype(str)
            )
        else:
            frames = cached_game_frames(ENGINE, cfg, ids)
            run_cols = ["nfl_id", "play_key"]
            frames["track_label"] = (
                frames["nfl_id"].map(insight_labels) + " · " + frames["play_label"]
            )

        if frames.empty:
            st.info("No tracking frames are available for this selection.")
        else:
            signals = {
                "Distance (yd)": "distance",
                "Speed (yd/s)": "s",
                "Acceleration (yd/s²)": "a",
                "Jerk (yd/s³)": "jerk",
            }
            metric_label = st.selectbox(
                "Time-series signal",
                list(signals),
                index=3,
                key="insight_timeseries_signal",
            )
            track_labels = frames["track_label"].dropna().unique().tolist()
            st.session_state["insight_tracks"] = [
                label for label in st.session_state.get("insight_tracks", [])
                if label in track_labels
            ][:8]
            selected_tracks = st.multiselect(
                "Runs / plays to plot (max 8)",
                track_labels,
                default=track_labels[: min(4, len(track_labels))],
                max_selections=8,
                key="insight_tracks",
            )
            chart_frames = frames[frames["track_label"].isin(selected_tracks)]
            if not chart_frames.empty:
                time_fig = px.line(
                    chart_frames,
                    x="t",
                    y=signals[metric_label],
                    color="track_label",
                    title=f"{metric_label} over time",
                    labels={"t": "Time (s)", signals[metric_label]: metric_label},
                )
                st.plotly_chart(time_fig, width="stretch")

            run_stats = (
                frames.groupby("track_label", as_index=False)
                .agg(
                    distance_yd=("distance", "max"),
                    mean_speed=("s", "mean"),
                    max_speed=("s", "max"),
                    mean_acceleration=("a", "mean"),
                    max_acceleration=("a", "max"),
                    mean_jerk=("jerk", "mean"),
                    jerk_std=("jerk", "std"),
                    max_abs_jerk=("jerk", lambda values: values.abs().max()),
                    jerk_samples=("jerk", "count"),
                )
            )
            st.subheader("Run / play statistics")
            st.dataframe(run_stats, hide_index=True, width="stretch")

            if st.session_state.get("insight_spectrum_track") not in track_labels:
                st.session_state.pop("insight_spectrum_track", None)
            spectrum_track = st.selectbox(
                "Track for spectrum",
                track_labels,
                key="insight_spectrum_track",
            )
            selected_frame = frames[frames["track_label"] == spectrum_track]
            spectrum_parts = []
            for signal_name, column in (
                ("Distance", "distance"),
                ("Speed", "s"),
                ("Acceleration", "a"),
                ("Jerk", "jerk"),
            ):
                spectrum = nd.spectrum_for_series(selected_frame["t"], selected_frame[column])
                if not spectrum.empty:
                    spectrum["signal"] = signal_name
                    spectrum_parts.append(spectrum)
            if spectrum_parts:
                spectrum_plot = pd.concat(spectrum_parts, ignore_index=True)
                spectrum_fig = px.line(
                    spectrum_plot,
                    x="frequency_hz",
                    y="power",
                    facet_col="signal",
                    facet_col_wrap=2,
                    title=f"Detrended power spectrum · {spectrum_track}",
                    labels={"frequency_hz": "Frequency (Hz)", "power": "Power"},
                )
                st.plotly_chart(spectrum_fig, width="stretch")
            else:
                st.info("At least 8 distinct valid samples are needed for a spectrum.")

            spectra = nd.tracking_spectrum_features(frames, run_cols)
            if not spectra.empty:
                player_spectra = spectra.groupby("nfl_id", as_index=False).mean(numeric_only=True)
                cohort = insight_candidates.merge(player_spectra, on="nfl_id", how="inner")
                spectrum_metrics = [
                    column for column in spectra.columns
                    if column.startswith("spectrum_") and cohort[column].notna().any()
                ]
                if spectrum_metrics and CAREER_COLS:
                    st.subheader("Spectral features vs career outcomes")
                    relation_cols = st.columns(3)
                    relation_metric = relation_cols[0].selectbox(
                        "Spectral feature", spectrum_metrics, key="insight_relation_metric"
                    )
                    relation_outcome = relation_cols[1].selectbox(
                        "Career outcome",
                        CAREER_COLS,
                        default_idx(CAREER_COLS, "success_index"),
                        key="insight_relation_outcome",
                    )
                    relation_method = relation_cols[2].radio(
                        "Correlation", ["spearman", "pearson"], horizontal=True, key="insight_relation_method"
                    )
                    relation_data = cohort.dropna(subset=[relation_metric, relation_outcome])
                    r, p, n = nd.corr_pair(
                        relation_data[relation_metric],
                        relation_data[relation_outcome],
                        relation_method,
                        min_n=5,
                    )
                    if n >= 5 and pd.notna(r):
                        st.caption(f"n={n} · {relation_method.title()} r={r:.3f} · p={p:.3g}")
                    else:
                        st.caption(f"n={n}; at least 5 non-constant player values are needed for correlation.")
                    relation_fig = px.scatter(
                        relation_data,
                        x=relation_metric,
                        y=relation_outcome,
                        color="pos_group",
                        hover_data=HOVER,
                        title=f"{relation_metric} vs {relation_outcome}",
                    )
                    st.plotly_chart(relation_fig, width="stretch")
                    st.caption(
                        "Spectral features are averaged across the selected player's runs/plays. "
                        "Small selected-player samples are descriptive and should not be interpreted causally."
                    )

# ---- SQL console ---------------------------------------------------------- #
with tab_sql:
    st.markdown("Read-only SQL (single `SELECT`/`WITH`). Tables: "
                "`players, games, player_play, player_career_successes, combine_results, combine_tracking, "
                "game_tracking_2023/2024/2025`.")
    with st.expander("Table & column browser (real database names)"):
        @st.cache_data(ttl=3600, show_spinner=False)
        def _schema(_engine, cfg):
            return nd.run_readonly(_engine, """SELECT table_name, column_name, data_type
                FROM information_schema.columns WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position""")
        sch = _schema(ENGINE, cfg)
        tname = st.selectbox("Table", sorted(sch["table_name"].unique()), key="sch_tbl")
        st.dataframe(sch[sch["table_name"] == tname][["column_name", "data_type"]], hide_index=True, width="stretch")
        st.caption("Note: in the app's own tables the position column is called `position`, "
                   "but in the database it is `players.nfl_position`; `combine_results.combine_height/weight` "
                   "appear as `height/weight` in the app.")
    default_sql = """SELECT p.nfl_position, count(*) AS n,
       round(avg(c.forty)::numeric, 3) AS avg_forty,
       round(avg(s.career_games_started)::numeric, 1) AS avg_starts
FROM players p
JOIN combine_results c USING (nfl_id)
JOIN player_career_successes s USING (nfl_id)
GROUP BY 1 ORDER BY n DESC"""
    st.session_state.setdefault("sql_text", default_sql)
    q_text = st.text_area("Query", key="sql_text", height=180)
    c1, c2 = st.columns([1, 4])
    lim = c2.number_input("Row limit", 10, 100000, 5000, step=500)
    if c1.button("Run", type="primary"):
        try:
            st.session_state["sql_res"] = nd.run_readonly(ENGINE, q_text, limit=int(lim))
        except Exception as e:  # noqa: BLE001
            st.session_state["sql_res"] = None
            st.error(str(e))
    res = st.session_state.get("sql_res")
    if res is not None:
        st.dataframe(res, width="stretch", hide_index=True)
        st.download_button("Download CSV", res.to_csv(index=False), "query.csv", "text/csv")
        num = res.select_dtypes("number").columns.tolist()
        if len(res.columns) >= 2 and num:
            with st.expander("Quick chart"):
                cc = st.columns(4)
                xx = cc[0].selectbox("X", res.columns, key="sx")
                yy = cc[1].selectbox("Y", num, key="sy")
                ct = cc[2].selectbox("Type", ["bar", "line", "scatter"], key="st")
                co = cc[3].selectbox("Color", [None] + list(res.columns), key="sc",
                                     format_func=lambda x: "—" if x is None else x)
                st.plotly_chart(getattr(px, ct)(res, x=xx, y=yy, color=co), width="stretch")

# ---- Data ----------------------------------------------------------------- #
with tab_data:
    st.markdown("Filtered master table (one row per player: draft, combine, tracking features, career).")
    show = st.multiselect("Columns", list(df.columns),
                          default=[c_ for c_ in ["display_name", "position", "draft_year", "draft_overall_pick",
                                                 "forty", "vertical", "weight", "trk40_vmax", "career_games_started",
                                                 "total_snaps", "success_index"] if c_ in df.columns])
    st.dataframe(df[show] if show else df, width="stretch", hide_index=True)
    st.download_button("Download filtered data (CSV)", df.to_csv(index=False), "nfl_master_filtered.csv", "text/csv")
    miss = df[FEATURE_COLS].isna().mean().sort_values(ascending=False).head(25)
    st.plotly_chart(px.bar(miss[::-1], orientation="h", title="Missing share by feature (top 25)")
                    .update_layout(showlegend=False, xaxis_title="", yaxis_title=""), width="stretch")