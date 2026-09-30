"""Evaluation boundaries for the chronological bike-sharing example."""
import csv
import io
import unittest

import numpy as np

from examples.bike_sharing import NAMES, cyclic_matrix, parse_hourly, split_indices


class BikeSharingProtocolTests(unittest.TestCase):
    def fixture(self):
        output = io.StringIO()
        columns = ['instant', 'dteday', *NAMES, 'casual', 'registered', 'cnt']
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for i, (day, hour) in enumerate([('2012-10-01', 0), ('2012-06-30', 23),
                                         ('2012-07-01', 0), ('2012-09-30', 23)]):
            row = {name: 1 for name in NAMES}
            row.update(instant=i, dteday=day, hr=hour, casual=1000+i,
                       registered=2000+i, cnt=3000+2*i)
            writer.writerow(row)
        return output.getvalue().encode()

    def test_target_components_and_identifiers_never_enter_features(self):
        X, y, dates = parse_hourly(self.fixture())
        self.assertEqual(X.shape, (4, 12))
        self.assertLess(X.max(), 24)
        self.assertTrue((y >= 3000).all())
        self.assertEqual(dates[0], '2012-06-30')
        self.assertFalse(set(NAMES) & {'cnt', 'casual', 'registered', 'instant', 'dteday'})

    def test_calendar_boundaries_keep_whole_days_out_of_earlier_splits(self):
        _, _, dates = parse_hourly(self.fixture())
        train, validation, test = split_indices(dates)
        self.assertEqual(dates[train].tolist(), ['2012-06-30'])
        self.assertEqual(dates[validation].tolist(), ['2012-07-01', '2012-09-30'])
        self.assertEqual(dates[test].tolist(), ['2012-10-01'])
        self.assertEqual(sorted(np.concatenate([train, validation, test])), list(range(4)))

    def test_manual_calendar_encoding_uses_true_periods(self):
        X = np.ones((3, len(NAMES)))
        X[:, NAMES.index('hr')] = [0, 24, 48]
        X[:, NAMES.index('mnth')] = [1, 13, 25]
        X[:, NAMES.index('weekday')] = [0, 7, 14]
        encoded = cyclic_matrix(X, interactions=True)
        np.testing.assert_allclose(encoded, np.repeat(encoded[:1], 3, axis=0), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
