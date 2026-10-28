"""Data access, feature engineering and statistics for the NFL Combine ↔ Career lab.

Everything here is plain pandas/numpy/sklearn so it can be tested without Streamlit.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #

def make_engine(host: str, port: int | str, dbname: str, user: str, password: str):
    url = URL.create(
        "postgresql+psycopg2",
        username=user or None,
        password=password or None,
        host=host,
        port=int(port),
        database=dbname,
    )
    return create_engine(url, pool_pre_ping=True)


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|do|vacuum|merge)\b",
    re.IGNORECASE,
)


def run_readonly(engine, sql: str, params: dict | None = None, limit: int | None = None) -> pd.DataFrame:
    """Run a single SELECT/WITH statement inside a READ ONLY transaction."""
    clean = sql.strip().rstrip(";").strip()
    if not clean:
        raise ValueError("Empty query.")
    if ";" in clean:
        raise ValueError("Only one statement at a time is allowed.")
    if not re.match(r"^(select|with|show|explain)\b", clean, re.IGNORECASE):
        raise ValueError("Only SELECT / WITH / SHOW / EXPLAIN statements are allowed.")
    if _FORBIDDEN.search(clean):
        raise ValueError("The query contains a keyword that is not allowed in read-only mode.")
    if limit and re.match(r"^(select|with)\b", clean, re.IGNORECASE) and not re.search(r"\blimit\b", clean, re.IGNORECASE):
        clean = f"SELECT * FROM ({clean}) AS _q LIMIT {int(limit)}"
    with engine.connect().execution_options(postgresql_readonly=True) as conn:
        return pd.read_sql(text(clean), conn, params=params)


# --------------------------------------------------------------------------- #
# Column catalogue
# --------------------------------------------------------------------------- #

NUM_RE = r"^-?[0-9]+(\.[0-9]+)?$"

COMBINE_RAW = [
    "height", "weight", "hand_size", "arm_length", "wing_span",
    "ten_yd_split", "forty", "vertical", "broad_jump", "three_cone",
    "short_shuttle", "bench_reps", "ngs_athleticism_score",
    "ngs_college_production_score", "ngs_final_score",
]
# metrics where a LOWER value is better (times)
LOWER_IS_BETTER = {"ten_yd_split", "forty", "three_cone", "short_shuttle"}
COMBINE_DERIVED = [
    "bmi", "speed_score", "burst", "agility_sum", "age_at_draft",
    "height_weight_ratio", "ape_index", "hand_arm_ratio",
]
COMBINE_PCT = [f"{c}_pct" for c in COMBINE_RAW if not c.startswith("ngs_")] + [
    "speed_score_pct", "burst_pct", "agility_sum_pct",
]
TRACK40 = [
    "trk40_vmax", "trk40_t_vmax", "trk40_amax", "trk40_v1s", "trk40_v2s",
    "trk40_t10", "trk40_t20", "trk40_t30", "trk40_t40",
    "trk40_split_10_20", "trk40_split_20_40", "trk40_speed_hold",
    "trk40_max_angle", "trk40_mean_speed", "trk40_min_speed",
    "trk40_mean_accel", "trk40_min_accel",
    "trk40_mean_jerk", "trk40_max_abs_jerk",
]
GAME_TRACK = [
    "game_max_speed", "game_mean_speed", "game_min_speed",
    "game_max_accel", "game_mean_accel", "game_min_accel",
    "game_max_angle", "game_max_play_distance", "game_mean_play_distance",
    "game_tracking_games", "game_tracking_plays",
    "game_mean_jerk", "game_max_abs_jerk",
]
DRAFT_NUM = ["draft_overall_pick", "draft_round", "draft_pick_within_round", "pick_filled", "log_pick", "undrafted"]
CAREER_RAW = [
    "career_offensive_snaps", "career_defensive_snaps", "career_special_teams_snaps",
    "career_games_active", "career_games_started", "ap_all_pro_1st_team",
    "ap_all_pro_2nd_team", "pro_bowl_original_ballot",
]
CAREER_DERIVED = [
    "total_snaps", "start_rate", "snaps_per_game", "games_started_per_season",
    "snaps_per_season", "games_active_per_season", "all_pro_any", "all_pro_pts",
    "seasons", "success_index",
]
CATEGORICAL = [
    "position", "pos_group", "draft_year", "draft_round", "college_conference",
    "college_name", "combine_position", "undrafted",
]

POS_GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL",
    "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
    "K": "ST", "P": "ST", "LS": "ST",
}

MASTER_SQL = f"""
SELECT
  p.nfl_id, p.display_name, p.draft_year, p.nfl_position AS position, p.birth_date,
  p.college_name, p.college_conference, p.draft_round, p.draft_pick_within_round,
  p.draft_overall_pick,
  c.combine_position,
  c.combine_height AS height, c.combine_weight AS weight, c.hand_size, c.arm_length, c.wing_span,
  c.ten_yd_split, c.forty, c.vertical, c.broad_jump,
  CASE WHEN c.three_cone    ~ '{NUM_RE}' THEN c.three_cone::numeric    END AS three_cone,
  CASE WHEN c.short_shuttle ~ '{NUM_RE}' THEN c.short_shuttle::numeric END AS short_shuttle,
  CASE WHEN c.bench_reps    ~ '{NUM_RE}' THEN c.bench_reps::numeric    END AS bench_reps,
  c.ngs_athleticism_score,
  CASE WHEN c.ngs_college_production_score ~ '{NUM_RE}' THEN c.ngs_college_production_score::numeric END AS ngs_college_production_score,
  CASE WHEN c.ngs_final_score ~ '{NUM_RE}' THEN c.ngs_final_score::numeric END AS ngs_final_score,
  s.career_offensive_snaps, s.career_defensive_snaps, s.career_special_teams_snaps,
  s.career_games_active, s.career_games_started,
  s.ap_all_pro_1st_team, s.ap_all_pro_2nd_team, s.pro_bowl_original_ballot
FROM players p
LEFT JOIN combine_results c ON c.nfl_id = p.nfl_id
LEFT JOIN player_career_successes s ON s.nfl_id = p.nfl_id
"""

_TEXT_COLS = {
    "display_name", "position", "college_name", "college_conference", "combine_position", "birth_date",
}

# Drill names that look like renamed duplicates across years. Merge them so a
# drill has one column per player. Edit if you disagree.
DRILL_ALIASES = {
    "PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL",
    "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",
    "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT",
    "LINE": "LINE_DRILL",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_master(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(MASTER_SQL), conn)
    return prepare_master(df)


def prepare_master(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and add derived combine / draft columns."""
    df = df.copy()
    for c in df.columns:
        if c not in _TEXT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["position"] = df["position"].fillna("UNK")
    df["pos_group"] = df["position"].map(POS_GROUP).fillna("OTHER")
    df["college_conference"] = df["college_conference"].fillna("Unknown")

    # draft features
    df["undrafted"] = df["draft_overall_pick"].isna().astype(int)
    fill = (df["draft_overall_pick"].max() if df["draft_overall_pick"].notna().any() else 260) + 1
    df["pick_filled"] = df["draft_overall_pick"].fillna(fill)
    df["log_pick"] = np.log(df["pick_filled"])
    df["draft_round"] = df["draft_round"].astype("float")

    # combine-derived
    df["bmi"] = df["weight"] * 703 / (df["height"] ** 2)
    df["speed_score"] = df["weight"] * 200 / (df["forty"] ** 4)
    df["burst"] = df["vertical"] + df["broad_jump"]
    df["agility_sum"] = df["three_cone"] + df["short_shuttle"]
    df["height_weight_ratio"] = df["height"] / df["weight"].where(df["weight"] > 0)
    df["ape_index"] = df["wing_span"] / df["height"].where(df["height"] > 0)
    df["hand_arm_ratio"] = df["hand_size"] / df["arm_length"].where(df["arm_length"] > 0)
    draft_date = pd.to_datetime(df["draft_year"].astype("Int64").astype(str) + "-04-25", errors="coerce")
    df["age_at_draft"] = (draft_date - df["birth_date"]).dt.days / 365.25

    # position-relative athletic percentiles (higher = better)
    for c in [c for c in COMBINE_RAW if not c.startswith("ngs_")] + ["speed_score", "burst", "agility_sum"]:
        df[f"{c}_pct"] = within_group_percentile(
            df, c, by="position", higher_is_better=c not in LOWER_IS_BETTER and c != "agility_sum"
        )

    # Focused movement translation metrics for scouting conversations.
    df["movement_translation_skill_index"] = movement_translation_index(df, focus="skill", by="pos_group")
    df["movement_translation_trench_index"] = movement_translation_index(df, focus="trench", by="pos_group")
    return df


def movement_translation_index(df: pd.DataFrame, focus: str = "skill", by: str = "pos_group") -> pd.Series:
    """Return a 0-100 movement-translation score for a player profile.

    Higher is better. The skill version rewards top-end speed, burst and agility for
    route/coverage types; the trench version rewards first-step burst and change-of-direction
    ability for line-of-scrimmage roles.
    """
    if focus not in {"skill", "trench"}:
        raise ValueError("focus must be 'skill' or 'trench'")
    if by not in df.columns:
        by = "position" if "position" in df.columns else None

    if focus == "skill":
        criteria = [
            ("trk40_vmax", True),
            ("vertical", True),
            ("broad_jump", True),
            ("three_cone", False),
            ("short_shuttle", False),
        ]
    else:
        criteria = [
            ("forty", False),
            ("vertical", True),
            ("broad_jump", True),
            ("three_cone", False),
            ("short_shuttle", False),
            ("bench_reps", True),
        ]

    parts = []
    for col, higher_is_better in criteria:
        if col not in df.columns:
            continue

        group_by = by if by is not None else None
        if group_by is not None and df.groupby(group_by).size().min() >= 5:
            part = within_group_percentile(df, col, by=group_by, higher_is_better=higher_is_better)
        else:
            r = df[col].rank(pct=True, method="average")
            if not higher_is_better:
                denom = max(1, int(df[col].notna().sum()))
                r = 1 - r + (1 / denom)
            part = (r * 100).clip(0, 100)
        parts.append(part)

    if not parts:
        raise ValueError(f"No movement columns found for focus='{focus}'")

    score = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return score.clip(0, 100)


def add_career_metrics(df: pd.DataFrame, last_season: int = 2025) -> pd.DataFrame:
    df = df.copy()
    df["total_snaps"] = df[["career_offensive_snaps", "career_defensive_snaps", "career_special_teams_snaps"]].sum(
        axis=1, min_count=1
    )
    df["seasons"] = (last_season - df["draft_year"] + 1).clip(lower=1)
    ga = df["career_games_active"].replace(0, np.nan)
    df["start_rate"] = df["career_games_started"] / ga
    df["snaps_per_game"] = df["total_snaps"] / ga
    df["games_started_per_season"] = df["career_games_started"] / df["seasons"]
    df["games_active_per_season"] = df["career_games_active"] / df["seasons"]
    df["snaps_per_season"] = df["total_snaps"] / df["seasons"]
    df["all_pro_any"] = df[["ap_all_pro_1st_team", "ap_all_pro_2nd_team"]].sum(axis=1, min_count=1)
    df["all_pro_pts"] = 2 * df["ap_all_pro_1st_team"].fillna(0) + df["ap_all_pro_2nd_team"].fillna(0)
    return df


def success_index(df: pd.DataFrame, metrics: list[str]) -> pd.Series:
    """Mean percentile (0-100) of the chosen career metrics within position × draft year."""
    if not metrics:
        return pd.Series(np.nan, index=df.index)
    pct = pd.concat(
        [df.groupby(["position", "draft_year"])[m].rank(pct=True, method="average") for m in metrics], axis=1
    )
    return pct.mean(axis=1) * 100


def within_group_percentile(df: pd.DataFrame, col: str, by: str = "position",
                            higher_is_better: bool = True, min_group: int = 5) -> pd.Series:
    if col not in df:
        return pd.Series(np.nan, index=df.index)
    r = df.groupby(by)[col].rank(pct=True, method="average")
    if not higher_is_better:
        r = 1 - r + (1 / df.groupby(by)[col].transform("count").clip(lower=1))
    n = df.groupby(by)[col].transform("count")
    r = r.where(n >= min_group)
    return (r * 100).clip(0, 100)


def load_tracking_summary(engine) -> pd.DataFrame:
    """One row per player × drill with aggregate features (PLAYER entities only)."""
    sql = """
    WITH tracking AS (
        SELECT nfl_id, drill_name, attempt, "time", s, a, dis, dir,
               LAG(a) OVER (
                   PARTITION BY nfl_id, drill_name, attempt ORDER BY "time"
               ) AS previous_accel,
               LAG("time") OVER (
                   PARTITION BY nfl_id, drill_name, attempt ORDER BY "time"
               ) AS previous_time
        FROM combine_tracking
        WHERE entity_type = 'PLAYER'
    ),
    deltas AS (
        SELECT *,
               (a - previous_accel) /
                   NULLIF(EXTRACT(EPOCH FROM ("time" - previous_time)), 0) AS jerk,
               dir - LAG(dir) OVER (
                   PARTITION BY nfl_id, drill_name, attempt ORDER BY "time"
               ) AS direction_delta
        FROM tracking
    ),
    frames AS (
        SELECT *,
               CASE WHEN ABS(direction_delta) > 180
                    THEN 360 - ABS(direction_delta)
                    ELSE ABS(direction_delta)
               END AS turn_angle
        FROM deltas
    )
    SELECT nfl_id, drill_name, attempt,
           MAX(s) AS max_speed, MIN(s) AS min_speed, AVG(s) AS mean_speed,
           MAX(a) AS max_accel, MIN(a) AS min_accel, AVG(a) AS mean_accel,
           AVG(jerk) AS mean_jerk, MAX(ABS(jerk)) AS max_abs_jerk,
           MAX(turn_angle) AS max_angle, SUM(dis) AS total_dist,
           EXTRACT(EPOCH FROM (MAX("time") - MIN("time"))) AS duration
    FROM frames
    GROUP BY nfl_id, drill_name, attempt
    """
    with engine.connect() as conn:
        raw = pd.read_sql(text(sql), conn)
    return summarize_tracking(raw)


def summarize_tracking(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["drill_name"] = raw["drill_name"].replace(DRILL_ALIASES)
    for c in ["max_speed", "min_speed", "mean_speed", "max_accel", "min_accel",
              "mean_accel", "mean_jerk", "max_abs_jerk", "max_angle", "total_dist", "duration"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    out = (
        raw.groupby(["nfl_id", "drill_name"])
        .agg(max_speed=("max_speed", "max"), max_accel=("max_accel", "max"),
             min_speed=("min_speed", "min"), mean_speed=("mean_speed", "mean"),
             min_accel=("min_accel", "min"), mean_accel=("mean_accel", "mean"),
             mean_jerk=("mean_jerk", "mean"), max_abs_jerk=("max_abs_jerk", "max"),
             max_angle=("max_angle", "max"),
             duration=("duration", "min"), total_dist=("total_dist", "mean"),
             attempts=("attempt", "nunique"))
        .reset_index()
    )
    return out


def pivot_drills(
    summary: pd.DataFrame,
    drills: list[str],
    feats=("max_speed", "max_accel", "max_angle", "mean_speed", "min_speed",
           "mean_accel", "min_accel", "mean_jerk", "max_abs_jerk", "duration", "total_dist"),
) -> pd.DataFrame:
    if not drills:
        return pd.DataFrame(columns=["nfl_id"])
    sub = summary[summary["drill_name"].isin(drills)]
    wide = sub.pivot(index="nfl_id", columns="drill_name", values=list(feats))
    wide.columns = [f"drill_{d}__{f}" for f, d in wide.columns]
    return wide.reset_index()


def load_game_tracking_summary(engine) -> pd.DataFrame:
    """Per-player game tracking metrics aggregated from play-level summaries."""
    sql = """
    WITH tracking AS (
        SELECT nfl_id, game_id, play_id, "time", s, a, dis, dir FROM game_tracking_2023
        UNION ALL
        SELECT nfl_id, game_id, play_id, "time", s, a, dis, dir FROM game_tracking_2024
        UNION ALL
        SELECT nfl_id, game_id, play_id, "time", s, a, dis, dir FROM game_tracking_2025
    ),
    direction_deltas AS (
        SELECT *,
               LAG(a) OVER (
                   PARTITION BY nfl_id, game_id, play_id ORDER BY "time"
               ) AS previous_accel,
               LAG("time") OVER (
                   PARTITION BY nfl_id, game_id, play_id ORDER BY "time"
               ) AS previous_time,
               dir - LAG(dir) OVER (
                   PARTITION BY nfl_id, game_id, play_id ORDER BY "time"
               ) AS direction_delta
        FROM tracking
    ),
    frames AS (
        SELECT *,
               (a - previous_accel) /
                   NULLIF(EXTRACT(EPOCH FROM ("time" - previous_time)), 0) AS jerk,
               CASE WHEN ABS(direction_delta) > 180
                    THEN 360 - ABS(direction_delta)
                    ELSE ABS(direction_delta)
               END AS turn_angle
        FROM direction_deltas
    )
    SELECT nfl_id, game_id, play_id,
           MAX(s) AS max_speed, MIN(s) AS min_speed, AVG(s) AS mean_speed,
           MAX(a) AS max_accel, MIN(a) AS min_accel, AVG(a) AS mean_accel,
           AVG(jerk) AS mean_jerk, MAX(ABS(jerk)) AS max_abs_jerk,
           MAX(turn_angle) AS max_angle, SUM(dis) AS play_distance
    FROM frames
    GROUP BY nfl_id, game_id, play_id
    """
    with engine.connect() as conn:
        play_metrics = pd.read_sql(text(sql), conn)
    return summarize_game_tracking(play_metrics)


def summarize_game_tracking(play_metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-play metrics to one row per player, weighting plays equally for means."""
    columns = ["nfl_id"] + GAME_TRACK
    if play_metrics.empty:
        return pd.DataFrame(columns=columns)
    numeric = [
        "max_speed", "min_speed", "mean_speed", "max_accel", "min_accel",
        "mean_accel", "mean_jerk", "max_abs_jerk", "max_angle", "play_distance",
    ]
    data = play_metrics.copy()
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    out = data.groupby("nfl_id").agg(
        game_max_speed=("max_speed", "max"),
        game_mean_speed=("mean_speed", "mean"),
        game_min_speed=("min_speed", "min"),
        game_max_accel=("max_accel", "max"),
        game_mean_accel=("mean_accel", "mean"),
        game_min_accel=("min_accel", "min"),
        game_mean_jerk=("mean_jerk", "mean"),
        game_max_abs_jerk=("max_abs_jerk", "max"),
        game_max_angle=("max_angle", "max"),
        game_max_play_distance=("play_distance", "max"),
        game_mean_play_distance=("play_distance", "mean"),
        game_tracking_games=("game_id", "nunique"),
        game_tracking_plays=("play_id", "count"),
    )
    out["game_tracking_games"] = out["game_tracking_games"].astype(int)
    out["game_tracking_plays"] = out["game_tracking_plays"].astype(int)
    return out.reset_index()[columns]


def load_forty_features(engine) -> pd.DataFrame:
    sql = """
    SELECT nfl_id, attempt, "time", s, a, dis, dir
    FROM combine_tracking
    WHERE entity_type = 'PLAYER' AND drill_name = 'FORTY_YARD_DASH'
    """
    with engine.connect() as conn:
        raw = pd.read_sql(text(sql), conn)
    return forty_features(raw)


def _one_forty(g: pd.DataFrame, start_speed: float = 1.0) -> dict | None:
    g = g.sort_values("time")
    tsec = (g["time"] - g["time"].iloc[0]).dt.total_seconds().to_numpy()
    s = pd.to_numeric(g["s"], errors="coerce").to_numpy(float)
    a = pd.to_numeric(g["a"], errors="coerce").to_numpy(float)
    d = np.nan_to_num(pd.to_numeric(g["dis"], errors="coerce").to_numpy(float))
    direction = pd.to_numeric(g.get("dir", pd.Series(np.nan, index=g.index)), errors="coerce").to_numpy(float)
    moving = np.where(s > start_speed)[0]
    if len(moving) < 5:
        return None
    i0 = moving[0]
    t, s, a, d, direction = tsec[i0:] - tsec[i0], s[i0:], a[i0:], d[i0:], direction[i0:]
    cum = np.cumsum(d)
    vmax_i = int(np.nanargmax(s))
    turn_angles = np.abs((np.diff(direction) + 180) % 360 - 180)
    finite_speed = s[np.isfinite(s)]
    finite_accel = a[np.isfinite(a)]
    finite_turn = turn_angles[np.isfinite(turn_angles)]
    jerk = np.gradient(a, t) if len(t) > 1 and np.all(np.diff(t) > 0) else np.full(len(a), np.nan)
    finite_jerk = jerk[np.isfinite(jerk)]
    out = {
        "trk40_vmax": float(s[vmax_i]),
        "trk40_t_vmax": float(t[vmax_i]),
        "trk40_amax": float(np.nanmax(a)),
        "trk40_min_speed": float(np.min(finite_speed)) if finite_speed.size else np.nan,
        "trk40_mean_speed": float(np.mean(finite_speed)) if finite_speed.size else np.nan,
        "trk40_min_accel": float(np.min(finite_accel)) if finite_accel.size else np.nan,
        "trk40_mean_accel": float(np.mean(finite_accel)) if finite_accel.size else np.nan,
        "trk40_mean_jerk": float(np.mean(finite_jerk)) if finite_jerk.size else np.nan,
        "trk40_max_abs_jerk": float(np.max(np.abs(finite_jerk))) if finite_jerk.size else np.nan,
        "trk40_max_angle": float(np.max(finite_turn)) if finite_turn.size else np.nan,
        "trk40_v1s": float(np.interp(1.0, t, s)) if t[-1] >= 1.0 else np.nan,
        "trk40_v2s": float(np.interp(2.0, t, s)) if t[-1] >= 2.0 else np.nan,
    }
    for yd in (10, 20, 30, 40):
        target = yd if yd < 40 else 39.5  # tracking can stop just short of the line
        if cum[-1] >= target:
            out[f"trk40_t{yd}"] = float(t[min(int(np.searchsorted(cum, target)), len(t) - 1)])
        else:
            out[f"trk40_t{yd}"] = np.nan
    out["trk40_split_10_20"] = out["trk40_t20"] - out["trk40_t10"]
    out["trk40_split_20_40"] = out["trk40_t40"] - out["trk40_t20"]
    mask = (cum >= 30) & (cum <= 40)
    out["trk40_speed_hold"] = float(np.nanmean(s[mask]) / out["trk40_vmax"]) if mask.any() else np.nan
    return out


def forty_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-player 40-yard-dash features from tracking, using each player's best attempt."""
    rows = []
    raw = raw.copy()
    raw["time"] = pd.to_datetime(raw["time"])
    for (nid, att), g in raw.groupby(["nfl_id", "attempt"]):
        f = _one_forty(g)
        if f:
            f.update(nfl_id=nid, attempt=att)
            rows.append(f)
    if not rows:
        return pd.DataFrame(columns=["nfl_id"] + TRACK40)
    df = pd.DataFrame(rows)
    # best attempt = fastest tracked 40 (fallback: highest top speed)
    df["_k"] = df["trk40_t40"].fillna(1e9) - df["trk40_vmax"] * 1e-3
    df = df.sort_values(["nfl_id", "_k"]).drop_duplicates("nfl_id")
    return df[["nfl_id"] + TRACK40]


def load_frames(engine, drill_name: str, nfl_ids: list[int]) -> pd.DataFrame:
    """Raw tracking frames for plotting (aliased drill names are all included)."""
    names = [drill_name] + [k for k, v in DRILL_ALIASES.items() if v == drill_name]
    sql = text(
        """
        SELECT nfl_id, drill_name, attempt, "time", x, y, s, a, dis, dir
        FROM combine_tracking
        WHERE entity_type = 'PLAYER' AND drill_name = ANY(:names) AND nfl_id = ANY(:ids)
        ORDER BY nfl_id, attempt, "time"
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"names": names, "ids": [int(i) for i in nfl_ids]})
    for c in ["x", "y", "s", "a", "dis", "dir"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    group_cols = ["nfl_id", "drill_name", "attempt"]
    df["t"] = df.groupby(group_cols)["time"].transform(lambda x: (x - x.min()).dt.total_seconds())
    df["distance"] = df.groupby(group_cols)["dis"].cumsum()
    return add_jerk(df, group_cols)


def add_jerk(frames: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Add acceleration's time derivative (yd/s³) independently within each track."""
    out = frames.copy()
    out["jerk"] = np.nan
    if out.empty or "t" not in out or "a" not in out:
        return out
    for _, group in out.groupby(group_cols, sort=False, dropna=False):
        t = pd.to_numeric(group["t"], errors="coerce").to_numpy(float)
        accel = pd.to_numeric(group["a"], errors="coerce").to_numpy(float)
        valid = np.isfinite(t) & np.isfinite(accel)
        valid_positions = np.flatnonzero(valid)
        if len(valid_positions) < 2:
            continue
        valid_times = t[valid]
        unique_times, unique_idx = np.unique(valid_times, return_index=True)
        if len(unique_times) < 2:
            continue
        unique_accel = accel[valid_positions[unique_idx]]
        jerk = np.gradient(unique_accel, unique_times)
        out.loc[group.index[valid_positions[unique_idx]], "jerk"] = jerk
    return out


def prepare_game_tracking_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Clean in-game tracking rows for a single player or a small group of players."""
    if df.empty:
        return df.copy()
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["nfl_id", "game_id", "play_id", "time"]).copy()
    for c in ["x", "y", "s", "a", "dis", "o", "dir"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values(["nfl_id", "game_id", "play_id", "time"]).reset_index(drop=True)
    out["play_key"] = out["game_id"].astype(str) + "_" + out["play_id"].astype(str)
    out["play_label"] = out["game_id"].astype(str) + " · play " + out["play_id"].astype(str)
    out["t"] = out.groupby(["nfl_id", "play_key"])["time"].transform(lambda x: (x - x.min()).dt.total_seconds())
    out["distance"] = out.groupby(["nfl_id", "play_key"])["dis"].cumsum()
    return add_jerk(out, ["nfl_id", "play_key"])


def load_game_tracking_frames(engine, nfl_ids: list[int]) -> pd.DataFrame:
    """Union all game tracking tables for the selected players and prepare them for plotting."""
    if not nfl_ids:
        return pd.DataFrame(columns=["nfl_id", "game_id", "play_id", "time", "x", "y", "s", "a", "dis", "o", "dir", "event"])
    ids = [int(i) for i in nfl_ids]
    sql = text(
        """
        SELECT nfl_id, game_id, play_id, time, x, y, s, a, dis, o, dir, event
        FROM game_tracking_2023
        WHERE nfl_id = ANY(:ids)
        UNION ALL
        SELECT nfl_id, game_id, play_id, time, x, y, s, a, dis, o, dir, event
        FROM game_tracking_2024
        WHERE nfl_id = ANY(:ids)
        UNION ALL
        SELECT nfl_id, game_id, play_id, time, x, y, s, a, dis, o, dir, event
        FROM game_tracking_2025
        WHERE nfl_id = ANY(:ids)
        ORDER BY nfl_id, game_id, play_id, time
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"ids": ids})
    return prepare_game_tracking_frames(df)


def spectrum_for_series(time: pd.Series, values: pd.Series, min_samples: int = 8) -> pd.DataFrame:
    """Return a detrended periodogram after resampling a track to its median frame interval."""
    t = pd.to_numeric(time, errors="coerce").to_numpy(float)
    y = pd.to_numeric(values, errors="coerce").to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]
    if len(t) < min_samples:
        return pd.DataFrame(columns=["frequency_hz", "power"])
    order = np.argsort(t)
    t, y = t[order], y[order]
    t, unique_idx = np.unique(t, return_index=True)
    y = y[unique_idx]
    if len(t) < min_samples:
        return pd.DataFrame(columns=["frequency_hz", "power"])
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return pd.DataFrame(columns=["frequency_hz", "power"])
    sample_times = np.arange(t[0], t[-1] + dt * 0.25, dt)
    if len(sample_times) < min_samples:
        return pd.DataFrame(columns=["frequency_hz", "power"])
    sampled = np.interp(sample_times, t, y)
    frequencies, power = signal.periodogram(
        sampled, fs=1 / dt, detrend="linear", scaling="spectrum"
    )
    return pd.DataFrame({"frequency_hz": frequencies[1:], "power": power[1:]})


def tracking_spectrum_features(
    frames: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """Summarize non-zero spectral peaks for distance, speed, acceleration and jerk per track."""
    feature_columns = [
        f"spectrum_{signal_name}_{metric}"
        for signal_name in ["distance", "s", "a", "jerk"]
        for metric in ["peak_hz", "centroid_hz", "peak_power"]
    ]
    records = []
    signals = {"distance": "distance", "s": "s", "a": "a", "jerk": "jerk"}
    if frames.empty:
        return pd.DataFrame(columns=group_cols + feature_columns)
    for keys, group in frames.groupby(group_cols, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_cols, keys))
        for name, column in signals.items():
            spectrum = spectrum_for_series(group["t"], group[column])
            if spectrum.empty:
                record[f"spectrum_{name}_peak_hz"] = np.nan
                record[f"spectrum_{name}_centroid_hz"] = np.nan
                record[f"spectrum_{name}_peak_power"] = np.nan
                continue
            frequencies = spectrum["frequency_hz"].to_numpy(float)
            power = spectrum["power"].to_numpy(float)
            peak_idx = int(np.argmax(power))
            total_power = float(np.sum(power))
            record[f"spectrum_{name}_peak_hz"] = float(frequencies[peak_idx])
            record[f"spectrum_{name}_centroid_hz"] = (
                float(np.sum(frequencies * power) / total_power) if total_power > 0 else np.nan
            )
            record[f"spectrum_{name}_peak_power"] = float(power[peak_idx])
        records.append(record)
    return pd.DataFrame(records, columns=group_cols + feature_columns)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def residualize(df: pd.DataFrame, cols: list[str], controls: list[str]) -> pd.DataFrame:
    """Return `cols` with the linear effect of `controls` removed (intercept included)."""
    out = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    ctrl = df[controls].astype(float)
    for c in cols:
        m = df[c].notna() & ctrl.notna().all(axis=1)
        if m.sum() < len(controls) + 3:
            continue
        X = np.column_stack([np.ones(m.sum()), ctrl[m].to_numpy()])
        y = df.loc[m, c].astype(float).to_numpy()
        if np.std(y) <= 1e-9 * max(1.0, abs(float(np.mean(y)))):
            continue  # constant feature: nothing to explain
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[m, c] = y - X @ beta
    return out


def _is_flat(v: pd.Series) -> bool:
    """True for constant series, including residuals that are only floating-point noise."""
    v = v.astype(float)
    if v.nunique() < 2:
        return True
    return float(np.std(v)) <= 1e-9 * max(1.0, abs(float(np.mean(v))))


def corr_pair(x: pd.Series, y: pd.Series, method: str = "pearson", min_n: int = 15):
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < min_n or _is_flat(x[m]) or _is_flat(y[m]):
        return np.nan, np.nan, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn = stats.spearmanr if method == "spearman" else stats.pearsonr
        r, p = fn(x[m].astype(float), y[m].astype(float))
    return float(r), float(p), n


def corr_table(df: pd.DataFrame, rows: list[str], cols: list[str], method="pearson",
               min_n=15, control: list[str] | None = None) -> pd.DataFrame:
    """Long table of correlations (pairwise complete) with n, p and a Benjamini-Hochberg q."""
    rows = [r for r in rows if r in df]
    cols = [c for c in cols if c in df]
    allc = list(dict.fromkeys(rows + cols))
    data = df[allc].copy()
    if control:
        data = residualize(df, allc, control)
    recs = []
    for r in rows:
        for c in cols:
            if r == c:
                continue
            rr, p, n = corr_pair(data[r], data[c], method, min_n)
            recs.append({"feature": r, "outcome": c, "r": rr, "p": p, "n": n})
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    out["q"] = bh_fdr(out["p"].to_numpy())
    return out


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty_like(ranked)
    res[order] = np.clip(ranked, 0, 1)
    q[ok] = res
    return q


def corr_matrix(tbl: pd.DataFrame, value="r") -> pd.DataFrame:
    return tbl.pivot(index="feature", columns="outcome", values=value)


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #

def cv_model(X: pd.DataFrame, y: pd.Series, kind: str = "rf", n_splits: int = 5,
             n_repeats_perm: int = 5, seed: int = 0) -> dict:
    """Cross-validated R², permutation importance and out-of-fold predictions."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline

    m = y.notna()
    X, y = X[m].reset_index(drop=True), y[m].reset_index(drop=True)
    idx = m[m].index

    def factory():
        if kind == "gb":
            return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                 min_samples_leaf=10, random_state=seed)
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=250, min_samples_leaf=3, max_features=0.5,
                                  n_jobs=-1, random_state=seed),
        )

    n_splits = max(2, min(n_splits, len(y) // 10 or 2))
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    r2s, imps = [], []
    for tr, te in kf.split(X):
        mdl = factory().fit(X.iloc[tr], y.iloc[tr])
        pred = mdl.predict(X.iloc[te])
        oof[te] = pred
        r2s.append(r2_score(y.iloc[te], pred))
        pi = permutation_importance(mdl, X.iloc[te], y.iloc[te], n_repeats=n_repeats_perm,
                                    random_state=seed, scoring="r2")
        imps.append(pi.importances_mean)
    return {
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "importance": pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values(ascending=False),
        "oof": pd.Series(oof, index=idx),
        "n": int(len(y)),
    }


def one_hot_position(df: pd.DataFrame, col: str = "pos_group") -> pd.DataFrame:
    return pd.get_dummies(df[col], prefix="pos", dtype=float)
