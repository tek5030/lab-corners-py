import numpy as np

from circle import Circle


def test_from_points():
    c = Circle.from_points((0, 5), (4, 1), (0, -3))
    np.testing.assert_almost_equal(c.center, np.asarray((0, 1)), 14)


def test_from_points_as_list():
    points = np.asarray(((0, 5), (4, 1), (0, -3)))
    c = Circle.from_points(*points)
    np.testing.assert_almost_equal(c.center, np.asarray((0, 1)), 14)


def test_distances():
    d = Circle.from_points((0, 5), (4, 1), (0, -3)).distances(((0, 6.5), (0, 5), (3, 1)))
    np.testing.assert_almost_equal(d, [1.5, 0, 1], 14)

