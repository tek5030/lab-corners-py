from typing import NamedTuple
from circle import Circle
import numpy as np


class CircleEstimate(NamedTuple):
    """Datatype for circle estimate as a result from CircleEstimator"""
    circle: Circle = Circle()
    num_iterations: int = 0
    num_inliers: int = 0
    is_inlier: np.ndarray = np.array([], dtype=bool)


def extract_inlier_points(estimate, points):
    return points[estimate.is_inlier]


class CircleEstimator:
    """A robust circle estimator based on circle point measurements"""

    def __init__(self, p = 0.99, distance_threshold = 5.0):
        """
        Constructs a circle estimator

        :param p: The desired probability of getting a good sample.
        :param distance_threshold: The maximum distance a good sample can have from the circle.
        """
        self._p = p
        self._distance_threshold = distance_threshold

    def estimate(self, points):
        """
        Estimates a circle based on the point measurements using RANSAC.

        :param points: Point measurements on the circle corrupted by noise.
        :return: The circle estimate based on the entire inlier set.
        """
        if points.shape[0] < 3:
            # Too few points to estimate any circle.
            return CircleEstimate()

        if len(points) == 3:
            # No need to estimate
            return CircleEstimate(circle=Circle.from_points(*points))

        # Estimate circle using RANSAC.
        estimate = self._ransac_estimator(points)

        if estimate.num_inliers == 0:
            return CircleEstimate()

        # Extract inlier points.
        inlier_pts = extract_inlier_points(estimate, points)

        # Estimate circle based on all the inliers.
        refined_circle = self._least_squares_estimator(inlier_pts)
        estimate._replace(circle=refined_circle)

        return estimate

    def _ransac_estimator(self, points):

        # Initialize maximum number of iterations.
        max_iterations = np.iinfo(np.int32).max

        # Perform RANSAC
        iterations = 0
        best_circle = Circle()
        best_num_inliers = 0
        best_is_inlier = np.array([], dtype=bool)

        while iterations < max_iterations:
            # Determine test circle by drawing minimal number of samples.
            test_circle = Circle.from_points(*points[np.random.choice(len(points), 3)])

            # Count number of inliers.
            is_inlier = test_circle.distances(points) < self._distance_threshold
            test_num_inliers = np.count_nonzero(is_inlier)

            # Check if this estimate gave a better result.
            # TODO 8: Remove break, and perform the correct test!
            if test_num_inliers > best_num_inliers:
                # Update circle with largest inlier set.
                best_num_inliers = test_num_inliers
                best_is_inlier = is_inlier
                best_circle = test_circle

                # Update max iterations.
                inlier_ratio = best_num_inliers / len(points)

                # FIXME: dele på 0
                max_iterations = int( np.log(1.0 - self._p) / np.log(1.0 - inlier_ratio*inlier_ratio*inlier_ratio))

            iterations += 1
            # break

        return CircleEstimate(
            circle=best_circle,
            num_iterations=iterations,
            num_inliers=best_num_inliers,
            is_inlier=best_is_inlier
        )

    def _least_squares_estimator(self, points):

        # Least-squares problem has the form A*p=b.
        # Construct A and b.
        a = np.c_[points, np.ones(len(points))]
        b = np.linalg.norm(points, axis=1)

        # Determine solution for p.
        p = np.linalg.lstsq(a, b)[0]

        # Extract center point and radius from the parameter vector p.
        center_point = 0.5 * p[:2]
        radius = np.sqrt(p[2] + np.linalg.norm(center_point))

        return Circle(center_point, radius)

if __name__ == "__main__":
    e = CircleEstimator()
    points = np.asarray(((0, 5), (4, 1), (0, -3)))
    c = e.estimate(points)
    print(f"estimate: {c.circle}")
