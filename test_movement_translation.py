import unittest

import pandas as pd

import nfl_data as nd


class MovementTranslationMetricTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
