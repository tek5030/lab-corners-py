from dataclasses import dataclass
from circle import Circle
import numpy as np


@dataclass
class CircleEstimate:
    """Datatype for circle estimate as a result from CircleEstimator"""
    circle: Circle = Circle()
    num_iterations: int = 0
    num_inliers: int = 0
    is_inlier: np.ndarray = np.array([], dtype=bool)


def extract_inlier_points(estimate, points):
    return points[estimate.is_inlier]


class CircleEstimator:
    """A robust circle estimator based on circle point measurements"""

    def __init__(self, p=0.99, distance_threshold=5.0):
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
        ransac_estimate = self._ransac_estimator(points)

        # Check if valid result.
        if ransac_estimate.num_inliers < 3:
            return None

        # Extract inlier points.
        inlier_pts = extract_inlier_points(ransac_estimate, points)

        # Estimate circle based on all the inliers.
        refined_circle = self._least_squares_estimator(inlier_pts)

        return CircleEstimate(
            circle=refined_circle,
            num_iterations=ransac_estimate.num_iterations,
            num_inliers=ransac_estimate.num_inliers,
            is_inlier=ransac_estimate.is_inlier
        )

    def _ransac_estimator(self, points):
        """Perform RANSAC estimation"""

        # Initialize maximum number of iterations.
        max_iterations = np.iinfo(np.int32).max

        # Perform RANSAC
        iterations = 0
        best_circle = Circle()
        best_num_inliers = 0
        best_is_inlier = np.array([], dtype=bool)

        while iterations < max_iterations:
            # Determine test circle by drawing minimal number of samples.
            test_circle = Circle.from_points(*points[np.random.choice(len(points), size=3, replace=False)])

            # Continue if the test circle was invalid.
            if not test_circle:
                continue

            # Count number of inliers.
            is_inlier = test_circle.distances(points) < self._distance_threshold
            test_num_inliers = np.count_nonzero(is_inlier)

            # Check if this estimate gave a better result.
            if test_num_inliers > best_num_inliers:
                # Update circle with largest inlier set.
                best_num_inliers = test_num_inliers
                best_is_inlier = is_inlier
                best_circle = test_circle

                # Update max iterations.
                inlier_ratio = best_num_inliers / len(points)
                max_iterations = int(np.log(1.0 - self._p) / np.log(1.0 - inlier_ratio*inlier_ratio*inlier_ratio))

            iterations += 1

        return CircleEstimate(
            circle=best_circle,
            num_iterations=iterations,
            num_inliers=best_num_inliers,
            is_inlier=best_is_inlier
        )

    def _least_squares_estimator(self, points):
        """ Estimates the least squares solution for the parameters of a circle given the points.

        The equations for the points (x_i, y_i) on the circle (x_c, y_c, r) is:
            (x_i - x_c)^2 + (y_i - y_c)^2 = r^2

        By multiplying out, we get the linear equations
            (2*x_c)*x_i + (2*y_c)*y_i + (r^2 - x_c^2 - y_x^2) = x_i^2 + y_i^2

        The least-squares problem then has the form A*p = b, where
            A = [x_i, y_i, 1],
            p = [2*x_c, 2*y_c, r^2 - x_c^2 - y_x^2]^T,
            b = [x_i^2 + y_i^2]

        by solving for p = [p_0, p_1, p_2], we get the following estimates for the circle parameters:
            x_c = 0.5 * p_0,
            y_c = 0.5 * p_1,
            r = sqrt(p_2 + x_c^2 + y_c^2)
        """
        # Construct A and b.
        A = np.c_[points, np.ones(len(points))]
        b = np.sum(np.square(points), axis=1)

        # Determine solution for p.
        p = np.linalg.lstsq(A, b, rcond=None)[0]

        # Extract center point and radius from the parameter vector p.
        center_point = 0.5 * p[:2]
        radius = np.sqrt(p[2] + np.sum(np.square(center_point)))

        return Circle(center_point, radius)
