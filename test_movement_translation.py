import unittest

import numpy as np
import pandas as pd

import nfl_data as nd


class MovementTranslationMetricTests(unittest.TestCase):
    def test_game_tracking_summary_aggregates_player_play_metrics(self):
        play_metrics = pd.DataFrame(
            {
                "nfl_id": [101, 101, 101, 202],
                "game_id": [1, 1, 2, 2],
                "play_id": [10, 11, 20, 30],
                "max_speed": [8.0, 10.0, 12.0, 9.0],
                "min_speed": [1.0, 0.0, 2.0, 1.0],
                "mean_speed": [4.0, 5.0, 8.0, 6.0],
                "max_accel": [2.0, 3.0, 4.0, 2.5],
                "min_accel": [-1.0, -2.0, -1.5, -1.0],
                "mean_accel": [0.5, 0.6, 0.7, 0.4],
                "mean_jerk": [1.0, 2.0, 3.0, 0.5],
                "max_abs_jerk": [4.0, 5.0, 6.0, 1.0],
                "max_angle": [30.0, 45.0, 60.0, 35.0],
                "play_distance": [20.0, 25.0, 30.0, 22.0],
            }
        )

        result = nd.summarize_game_tracking(play_metrics).set_index("nfl_id")

        self.assertEqual(result.loc[101, "game_max_speed"], 12.0)
        self.assertAlmostEqual(result.loc[101, "game_mean_speed"], (4 + 5 + 8) / 3)
        self.assertEqual(result.loc[101, "game_min_speed"], 0.0)
        self.assertEqual(result.loc[101, "game_max_accel"], 4.0)
        self.assertAlmostEqual(result.loc[101, "game_mean_accel"], (0.5 + 0.6 + 0.7) / 3)
        self.assertEqual(result.loc[101, "game_min_accel"], -2.0)
        self.assertAlmostEqual(result.loc[101, "game_mean_jerk"], 2.0)
        self.assertEqual(result.loc[101, "game_max_abs_jerk"], 6.0)
        self.assertEqual(result.loc[101, "game_max_angle"], 60.0)
        self.assertEqual(result.loc[101, "game_max_play_distance"], 30.0)
        self.assertAlmostEqual(result.loc[101, "game_mean_play_distance"], 25.0)
        self.assertEqual(result.loc[101, "game_tracking_games"], 2)
        self.assertEqual(result.loc[101, "game_tracking_plays"], 3)

    def test_prepare_master_adds_requested_combine_ratios(self):
        df = pd.DataFrame(
            {
                "nfl_id": [101],
                "display_name": ["Test Player"],
                "draft_year": [2025],
                "position": ["WR"],
                "birth_date": ["2002-01-01"],
                "college_name": ["Test U"],
                "college_conference": ["Test Conference"],
                "draft_round": [1],
                "draft_pick_within_round": [1],
                "draft_overall_pick": [1],
                "combine_position": ["WR"],
                "height": [72.0],
                "weight": [180.0],
                "hand_size": [9.0],
                "arm_length": [30.0],
                "wing_span": [74.0],
                "ten_yd_split": [1.55],
                "forty": [4.5],
                "vertical": [36.0],
                "broad_jump": [120.0],
                "three_cone": [6.8],
                "short_shuttle": [4.2],
                "bench_reps": [12],
                "trk40_vmax": [21.0],
            }
        )

        result = nd.prepare_master(df).iloc[0]

        self.assertAlmostEqual(result["bmi"], 180 * 703 / 72**2)
        self.assertAlmostEqual(result["height_weight_ratio"], 72 / 180)
        self.assertAlmostEqual(result["ape_index"], 74 / 72)
        self.assertAlmostEqual(result["hand_arm_ratio"], 9 / 30)

    def test_forty_features_use_circular_turn_angle_and_tracking_summaries(self):
        frames = pd.DataFrame(
            {
                "time": pd.date_range("2025-01-01", periods=10, freq="100ms"),
                "s": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "dis": [1.0] * 10,
                "dir": [359, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            }
        )

        result = nd._one_forty(frames)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["trk40_max_angle"], 2.0)
        self.assertAlmostEqual(result["trk40_mean_speed"], frames["s"].mean())
        self.assertAlmostEqual(result["trk40_mean_accel"], frames["a"].mean())
        self.assertAlmostEqual(result["trk40_min_speed"], frames["s"].min())
        self.assertAlmostEqual(result["trk40_min_accel"], frames["a"].min())
        self.assertAlmostEqual(result["trk40_mean_jerk"], 1.0)
        self.assertAlmostEqual(result["trk40_max_abs_jerk"], 1.0)

    def test_metric_ranks_faster_skill_players_higher(self):
        df = pd.DataFrame(
            {
                "position": ["WR", "WR", "DB", "DB"],
                "pos_group": ["WR", "WR", "DB", "DB"],
                "trk40_vmax": [22.0, 18.5, 21.0, 19.8],
                "three_cone": [6.4, 7.1, 6.8, 7.2],
                "short_shuttle": [4.1, 4.7, 4.3, 4.9],
                "success_index": [80.0, 40.0, 72.0, 30.0],
            }
        )

        result = nd.movement_translation_index(df, focus="skill")

        self.assertEqual(len(result), len(df))
        self.assertTrue(result.iloc[0] > result.iloc[1])
        self.assertTrue(result.iloc[2] > result.iloc[3])
        self.assertTrue(result.notna().all())
        self.assertTrue((0 <= result).all() and (result <= 100).all())

    def test_prepare_game_tracking_frames_adds_play_key_and_elapsed_time(self):
        df = pd.DataFrame(
            {
                "nfl_id": [101, 101, 101, 101],
                "game_id": [91, 91, 92, 92],
                "play_id": [100, 100, 200, 200],
                "time": pd.to_datetime([
                    "2024-01-01 00:00:02",
                    "2024-01-01 00:00:01",
                    "2024-01-02 00:00:05",
                    "2024-01-02 00:00:04",
                ]),
                "x": [10.0, 8.0, 11.0, 9.0],
                "y": [5.0, 4.0, 6.0, 5.0],
                "s": [3.0, 2.0, 4.0, 3.0],
                "a": [0.5, 0.4, 0.8, 0.7],
                "dis": [1.0, 0.5, 2.5, 1.5],
                "o": [120.0, 90.0, 180.0, 150.0],
                "dir": [45.0, 50.0, 110.0, 100.0],
                "event": ["ball_snap", "run", "ball_snap", "run"],
            }
        )

        result = nd.prepare_game_tracking_frames(df)

        self.assertIn("play_key", result.columns)
        self.assertEqual(result["play_key"].nunique(), 2)
        self.assertTrue((result["t"] >= 0).all())
        self.assertTrue(result.sort_values(["game_id", "play_id", "time"]).equals(result))
        first_play = result[result["game_id"] == 91]
        self.assertEqual(first_play["distance"].tolist(), [0.5, 1.5])
        self.assertEqual(first_play["jerk"].tolist(), [0.1, 0.1])

    def test_add_jerk_uses_elapsed_time_and_resets_at_track_boundaries(self):
        frames = pd.DataFrame(
            {
                "track": ["a", "a", "a", "b", "b", "b"],
                "t": [0.0, 1.0, 3.0, 0.0, 0.5, 1.5],
                "a": [0.0, 2.0, 6.0, 10.0, 9.0, 7.0],
            }
        )

        result = nd.add_jerk(frames, ["track"])

        self.assertEqual(result.loc[:2, "jerk"].tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(result.loc[3:, "jerk"].tolist(), [-2.0, -2.0, -2.0])

    def test_spectrum_features_find_known_frequency(self):
        sample_rate = 10
        times = pd.Series(np.arange(40) / sample_rate)
        values = pd.Series(np.sin(2 * np.pi * times))

        spectrum = nd.spectrum_for_series(times, values)

        self.assertFalse(spectrum.empty)
        peak_frequency = spectrum.loc[spectrum["power"].idxmax(), "frequency_hz"]
        self.assertAlmostEqual(peak_frequency, 1.0)


if __name__ == "__main__":
    unittest.main()
