import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybaseball as pyb
import seaborn as sns

from pybaseball import playerid_reverse_lookup
from cse_163_final import data_setup, get_clutch_rating, clutch_teams
import unittest


class TestDataProcessing(unittest.TestCase):
       
    def setUp(self):
        self._subset = pyb.statcast(start_dt='2022-03-31', end_dt='2022-05-30')
        self._subset['year'] = 2022
        self._events_to_keep = ["strikeout", "single", "double", "triple", "walk", "home_run", "field_out"]
        self._subset = self._subset[self._subset['events'].isin(self._events_to_keep)]


    def test_data_setup(self):
        clutch_df = self._subset[(self._subset['inning'] >= 8) & (abs(self._subset['home_score'] - self._subset['away_score']) <= 2)]
        nonclutch_df = self._subset

        # Edge case: Test for empty sets
        self.assertTrue(len(clutch_df) >= 0, "Clutch dataframe can be empty but shouldn't be negative.")
        self.assertTrue(len(nonclutch_df) >= 0, "Non-clutch dataframe can be empty but shouldn't be negative.")

        expected_clutch_length = len(clutch_df)
        expected_nonclutch_length = len(nonclutch_df)

        data_after_setup_true = data_setup(self._subset, True)
        data_after_setup_false = data_setup(self._subset, False)

        self.assertEqual(expected_clutch_length, len(data_after_setup_true))
        self.assertEqual(expected_nonclutch_length, len(data_after_setup_false))
        

    def test_get_clutch_rating(self):
        data_after_setup_true = data_setup(self._subset, True)
        data_after_setup_false = data_setup(self._subset, False)

        clutch_rating_data_true = get_clutch_rating(data_after_setup_true)
        clutch_rating_data_false = get_clutch_rating(data_after_setup_false)

        # Edge case: Test for empty sets
        self.assertFalse(clutch_rating_data_true.empty, "The clutch rating DataFrame (clutch) should not be empty.")
        self.assertFalse(clutch_rating_data_false.empty, "The clutch rating DataFrame (non-clutch) should not be empty.")

        self.assertEqual(560, len(clutch_rating_data_true))
        self.assertEqual(679, len(clutch_rating_data_false))

    def test_clutch_teams(self):
        data_after_setup_true = data_setup(self._subset, True)
        data_after_setup_false = data_setup(self._subset, False)

        clutch_rating_data_true = get_clutch_rating(data_after_setup_true)
        clutch_rating_data_false = get_clutch_rating(data_after_setup_false)

        clutch_teams_true = clutch_teams(clutch_rating_data_true)
        clutch_teams_false = clutch_teams(clutch_rating_data_false)

        # Edge case: Test for empty sets 
        self.assertFalse(clutch_teams_true, "The clutch rating DataFrame (clutch) should not be empty.")
        self.assertFalse(clutch_teams_false, "The clutch rating DataFrame (non-clutch) should not be empty.")

        #Since there are only 30 teams in MLB we should only see 30 
        self.assertEqual(30, len(clutch_teams_true))
        self.assertEqual(30, len(clutch_teams_false))

    def test_histogram_data_processing(self):
        data_after_setup_true = data_setup(self._subset, True)
        data_after_setup_false = data_setup(self._subset, False)
        
        clutch_rating_data_true = get_clutch_rating(data_after_setup_true)
        clutch_rating_data_false = get_clutch_rating(data_after_setup_false)

        not_clutch_average_rating = clutch_rating_data_false.groupby('full_name')['scaled_total_clutch_rating'].mean().reset_index()
        clutch_average_rating = clutch_rating_data_true.groupby('full_name')['scaled_total_clutch_rating'].mean().reset_index()

        average_rating_merged = pd.merge(not_clutch_average_rating, clutch_average_rating, on = ['full_name'], how = 'inner')
        average_rating_merged['clutch_above_normal'] = average_rating_merged['scaled_total_clutch_rating_x'] - average_rating_merged['scaled_total_clutch_rating_y']

        average_rating_merged = average_rating_merged.sort_values(by='clutch_above_normal', ascending=False)
        average_rating_merged = average_rating_merged[['full_name', 'clutch_above_normal']]

        self.assertFalse(average_rating_merged.empty, "The merged average ratings data frame should not be empty.")
        self.assertTrue(all(average_rating_merged['clutch_above_normal'].notnull()), "There should be no null values in the clutch_above_normal column.")
        

if __name__ == '__main__':
    unittest.main()


