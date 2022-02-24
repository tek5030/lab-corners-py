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
    def from_points(cls, point1, point2, point3):
        """
        Constructs a circle from three points on the circle.

        :param point1: First point on the circle.
        :param point2: Second point on the circle.
        :param point3: Third point on the circle.
        :return:
        """
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        point3 = np.asarray(point3)

        def homogeneous(x):
            return np.append(x, [1], axis=0)

        def hnormalized(x):
            return x[:-1] / x[-1]

        def center_line(p1, p2):
            m1 = 0.5 * (p1 + p2)
            q1 = (m1[0] + p2[1] - p1[1],
                  m1[1] - p2[0] + p1[0])
            return np.cross(homogeneous(m1), homogeneous(q1))

        # Compute homogeneous center lines.
        line_1 = center_line(point1, point2)
        line_2 = center_line(point2, point3)

        # Compute homogenous center point.
        center_h = np.cross(line_1, line_2)

        # Co-linear points will result in a center at infinity,
        # so return invalid circle in that case.
        if center_h[-1] == 0:
            return None

        # Construct circle.
        center = hnormalized(center_h)
        radius = np.linalg.norm(point1 - center)
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

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points.shape {points.shape} != nx2")

        return np.abs(np.linalg.norm(points - self._center, axis=1) - self._radius)

    def __str__(self):
        return f"center: {self._center}, radius: {self._radius}"
