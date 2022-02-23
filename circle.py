import numpy as np


class Circle:
    def __init__(self, center=(0., 0.), radius=0.0):
        """
        Construct a circle from a center point and a radius.

        :param center: The center point of the circle.
        :param radius: The radius of the circle.
        """
        self._center = center
        self._radius = radius

    @classmethod
    def from_points(cls, p1, p2, p3):
        """
        Constructs a circle from three points on the circle.

        :param p1: First point on the circle.
        :param p2: Second point on the circle.
        :param p3: Third point on the circle.
        :return:
        """
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        p3 = np.asarray(p3)

        homogeneous = lambda x: np.append(x, [1], axis=0)
        hnormalized = lambda x: x[:-1] / x[-1]  # fixme dele på 0

        # p1 and p2 define line_1 as their center line
        m1 = 0.5 * (p1 + p2)
        q1 = (m1[0] + p2[1] - p1[1],
              m1[1] - p2[0] + p1[0])
        line_1 = np.cross(homogeneous(m1), homogeneous(q1))

        #  p2 and p3 define the line_2 as their center line
        m2 = 0.5 * (p2 + p3)
        q2 = (m2[0] + p3[1] - p2[1],
              m2[1] - p3[0] + p2[0])
        line_2 = np.cross(homogeneous(m2), homogeneous(q2))
        #
        #  Determine circle
        center = hnormalized(np.cross(line_1, line_2))
        radius = np.linalg.norm(p1 - center)
        return cls(center, radius)

    @property
    def center(self):
        """
        The center point of the circle.
        :return: The center point.
        """
        return self._center

    @property
    def radius(self):
        """
        The circle radius.
        :return: The radius.
        """
        return self._radius

    def distances(self, points):
        """
        The distance from the circle to each given point.
        :param points: A nx2 set of points
        :return: The distances.
        """
        points = np.asarray(points)

        if points.ndim == 1:
            points = points[np.newaxis, :]
        #
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points.shape {points.shape} != nx2")

        return np.abs(np.linalg.norm(points - self._center, axis=1) - self._radius)

    def __str__(self):
        return f"center: {self._center}, radius: {self._radius}"
