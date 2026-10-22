-- PostgreSQL table creation script for the CSV files in this workspace
-- Table names match the CSV filenames without the .csv extension.

DROP TABLE IF EXISTS combine_results CASCADE;
CREATE TABLE combine_results (
    draft_year INTEGER,
    nfl_id BIGINT,
    combine_position TEXT,
    combine_height NUMERIC(6,3),
    combine_weight INTEGER,
    hand_size NUMERIC(4,2),
    arm_length NUMERIC(5,2),
    wing_span NUMERIC(5,2),
    ten_yd_split NUMERIC(4,3),
    forty NUMERIC(4,2),
    vertical NUMERIC(4,1),
    broad_jump INTEGER,
    three_cone TEXT,
    short_shuttle TEXT,
    bench_reps TEXT,
    ngs_athleticism_score INTEGER,
    ngs_college_production_score TEXT,
    ngs_final_score TEXT
);

DROP TABLE IF EXISTS combine_tracking CASCADE;
CREATE TABLE combine_tracking (
    draft_year INTEGER,
    event_id TEXT,
    nfl_id BIGINT,
    entity_type TEXT,
    time TIMESTAMP,
    drill_type TEXT,
    drill_name TEXT,
    attempt INTEGER,
    x NUMERIC(12,8),
    y NUMERIC(12,8),
    s NUMERIC(12,8),
    a NUMERIC(12,8),
    dis NUMERIC(12,8),
    dir NUMERIC(12,8)
);

DROP TABLE IF EXISTS game_tracking_2023 CASCADE;
CREATE TABLE game_tracking_2023 (
    game_id BIGINT,
    play_id INTEGER,
    nfl_id BIGINT,
    time TIMESTAMP,
    x NUMERIC(6,2),
    y NUMERIC(6,2),
    s NUMERIC(6,2),
    a NUMERIC(6,2),
    dis NUMERIC(6,2),
    o NUMERIC(6,2),
    dir NUMERIC(6,2),
    event TEXT
);

DROP TABLE IF EXISTS game_tracking_2024 CASCADE;
CREATE TABLE game_tracking_2024 (
    game_id BIGINT,
    play_id INTEGER,
    nfl_id BIGINT,
    time TIMESTAMP,
    x NUMERIC(6,2),
    y NUMERIC(6,2),
    s NUMERIC(6,2),
    a NUMERIC(6,2),
    dis NUMERIC(6,2),
    o NUMERIC(6,2),
    dir NUMERIC(6,2),
    event TEXT
);

DROP TABLE IF EXISTS game_tracking_2025 CASCADE;
CREATE TABLE game_tracking_2025 (
    game_id BIGINT,
    play_id INTEGER,
    nfl_id BIGINT,
    time TIMESTAMP,
    x NUMERIC(6,2),
    y NUMERIC(6,2),
    s NUMERIC(6,2),
    a NUMERIC(6,2),
    dis NUMERIC(6,2),
    o NUMERIC(6,2),
    dir NUMERIC(6,2),
    event TEXT
);

DROP TABLE IF EXISTS games CASCADE;
CREATE TABLE games (
    game_id BIGINT,
    game_key BIGINT,
    season INTEGER,
    season_type TEXT,
    week INTEGER,
    home_team_abbr TEXT,
    visitor_team_abbr TEXT
);

DROP TABLE IF EXISTS player_career_successes CASCADE;
CREATE TABLE player_career_successes (
    nfl_id BIGINT,
    career_offensive_snaps INTEGER,
    career_defensive_snaps INTEGER,
    career_special_teams_snaps INTEGER,
    career_games_active INTEGER,
    career_games_started INTEGER,
    ap_all_pro_1st_team INTEGER,
    ap_all_pro_2nd_team INTEGER,
    pro_bowl_original_ballot INTEGER
);

DROP TABLE IF EXISTS player_play CASCADE;
CREATE TABLE player_play (
    game_id BIGINT,
    play_id INTEGER,
    team_abbr TEXT,
    nfl_id BIGINT,
    lined_up_position TEXT,
    play_description TEXT,
    quarter INTEGER,
    down INTEGER,
    yards_to_go INTEGER,
    possession_team TEXT,
    yardline_side TEXT,
    yardline_number INTEGER,
    game_clock TEXT,
    pre_snap_home_score INTEGER,
    pre_snap_visitor_score INTEGER,
    play_direction TEXT,
    pre_snap_home_team_win_probability NUMERIC(10,8),
    pre_snap_visitor_team_win_probability NUMERIC(10,8),
    home_team_win_probability_added NUMERIC(10,8),
    visitor_team_win_probability_added NUMERIC(10,8),
    expected_points_added NUMERIC(12,8),
    expected_points NUMERIC(12,8),
    offense_formation TEXT,
    receiver_alignment TEXT,
    pass_result TEXT,
    pass_length INTEGER,
    play_nullified_by_penalty TEXT,
    penalty_yards INTEGER,
    pre_penalty_yards_gained INTEGER,
    yards_gained INTEGER,
    team_coverage_man_zone TEXT,
    team_coverage_type TEXT,
    in_motion_at_ball_snap TEXT,
    target TEXT,
    rec_yards INTEGER,
    rush_yards INTEGER,
    yards_after_catch NUMERIC(10,4),
    fumble TEXT,
    fumble_lost TEXT,
    route_ran TEXT,
    separation_at_pass_forward NUMERIC(10,6),
    cushion NUMERIC(10,6),
    expected_yards_after_catch NUMERIC(10,6),
    dropback_duration NUMERIC(10,6),
    extended_sack_allowed TEXT,
    pass_rushers_encountered INTEGER,
    peak_pressure_probability_allowed NUMERIC(10,6),
    pressure_allowed INTEGER,
    sack_allowed INTEGER,
    time_to_pressure_allowed NUMERIC(10,6),
    quick_pressure TEXT,
    unblocked_pressure TEXT,
    blitzing TEXT,
    player_get_off TEXT,
    sack INTEGER,
    tackle INTEGER,
    assist INTEGER,
    caused_forced_fumble INTEGER,
    recovered_fumble INTEGER,
    tackle_for_loss INTEGER,
    time_to_pressure NUMERIC(10,6),
    time_to_qb_hurry NUMERIC(10,6),
    coverage_assignment TEXT,
    coverage_assignment_at_snap TEXT
);

DROP TABLE IF EXISTS players CASCADE;
CREATE TABLE players (
    nfl_id BIGINT,
    display_name TEXT,
    draft_year INTEGER,
    nfl_position TEXT,
    birth_date DATE,
    college_name TEXT,
    college_conference TEXT,
    draft_round INTEGER,
    draft_pick_within_round INTEGER,
    draft_overall_pick INTEGER
);
