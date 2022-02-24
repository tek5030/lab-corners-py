import numpy as np

from solution_corners import CircleEstimator


def test_correct_result_for_minimal_problem():
    e = CircleEstimator()
    points = np.asarray(((0, 5), (4, 1), (0, -3)))
    c = e.estimate(points)

    np.testing.assert_almost_equal(c.circle.center, [0.0, 1.0])
    np.testing.assert_almost_equal(c.circle.radius, 4.0)
