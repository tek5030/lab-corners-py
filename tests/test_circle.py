import unittest

import numpy as np

from circle import Circle

class TestCircle(unittest.TestCase):
    def test_from_points(self):
        """Test that a circle can be created from a set of points"""
        c = Circle.from_points((0, 5), (4, 1), (0, -3))
        self.assertTrue(np.allclose(c.center, np.asarray((0, 1))))

    def test_from_points_as_list(self):
        points = np.asarray(((0, 5), (4, 1), (0, -3)))
        c = Circle.from_points(*points)
        self.assertTrue(np.allclose(c.center, np.asarray((0, 1))))

    def test_distances(self):
        d = Circle.from_points((0, 5), (4, 1), (0, -3)).distances(((0, 6.5), (0, 5), (3, 1)))
        self.assertTrue(np.allclose(d, [1.5, 0, 1]))

if __name__ == '__main__':
    unittest.main()
