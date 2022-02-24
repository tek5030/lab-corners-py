import cv2
import numpy as np
import timeit

from common_lab_utils import Circle, CircleEstimate,\
    create_1d_gaussian_kernel, extract_inlier_points, retain_best,\
    draw_circle_result, draw_corner_result


def run_corners_solution():
    # Connect to the camera.
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source {video_source}")
        return
    else:
        print(f"Successfully opened video source {video_source}")

    # Create window
    window_name = 'Lab: Estimating circles from corners'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Construct the corner detector.
    # Play around with the parameters!
    # When the second argument is true, additional debug visualizations are shown.
    det = CornerDetector(metric_type='harris', visualize=False)

    # Construct the circle estimator
    estimator = CircleEstimator()

    while True:
        # Read next frame.
        success, frame = cap.read()
        if not success:
            print(f"The video source {video_source} stopped")
            break

        # Convert frame to gray scale image.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform corner detection
        # Measure how long the processing takes.
        start = timeit.default_timer()

        keypoints, points = det.detect(gray_frame)

        end = timeit.default_timer()
        duration_corners = end - start

        # Keep the highest scoring points.
        best = retain_best(keypoints, 1000)
        keypoints = keypoints[best]
        points = points[best]

        # Estimate circle based on detected corner points
        start = timeit.default_timer()

        circle_estimate = estimator.estimate(points)

        end = timeit.default_timer()
        duration_circle = end - start

        # Show the results
        draw_corner_result(frame, keypoints, duration_corners)
        draw_circle_result(frame, keypoints, circle_estimate, duration_circle)
        cv2.imshow(window_name, frame)

        # Update the GUI and wait a short time for input from the keyboard.
        key = cv2.waitKey(1)

        # React to keyboard commands.
        if key == ord('q'):
            print("Quitting")
            break

    # Stop video source.
    cv2.destroyAllWindows()
    cap.release()


def create_1d_derivated_gaussian_kernel(sigma, radius=0):
    """
    Creates a Nx1 derivated gaussian filter kernel.

    :param sigma: The sigma (standard deviation) parameter for the gaussian.
    :param radius: The filter radius, so that N = 2*radius + 1. If set to 0, the radius will be computed so that radius = 3.5 * sigma.
    :return:
    """
    if radius <= 0:
        radius = int(np.ceil(3.5 * sigma))

    # TODO 1: Use create1DGaussianKernel to compute the derivated kernel.
    kernel, x = create_1d_gaussian_kernel(sigma, radius)

    kernel = kernel * x * (-1. / (sigma * sigma))

    return kernel, x


class CornerDetector:
    """A homebrewed corner detector!"""

    def __init__(self, metric_type, visualize=False, quality_level=0.01, gradient_sigma=1.0, window_sigma=2.0):
        """
        Constructs the corner detector.

        :param metric_type: The metric used to extract corners.
        :param visualize: Shows additional debug/educational figures when true.
        :param quality_level: The quality level used for thresholding areas with corners.
        :param gradient_sigma: The standard deviation for the gradient filter.
        :param window_sigma: The standard deviation for the window filters.
        """

        self._metric_type = metric_type
        self._visualize = visualize
        self._quality_level = quality_level
        self._window_sigma = window_sigma
        self._g_kernel = create_1d_gaussian_kernel(gradient_sigma)[0]
        self._dg_kernel = create_1d_derivated_gaussian_kernel(gradient_sigma)[0]
        self._win_kernel = create_1d_gaussian_kernel(window_sigma)[0]

    def detect(self, image):
        """
        Detects corners in an image.

        :param image: The image that is queried for corners.
        :return: An array of corner key points.
        """

        # TODO 2: Estimate image gradients Ix and Iy by combining _g_kernel and _dg_kernel.
        ix = cv2.sepFilter2D(image, cv2.CV_32F, self._dg_kernel, self._g_kernel)
        iy = cv2.sepFilter2D(image, cv2.CV_32F, self._g_kernel, self._dg_kernel)

        # TODO 3.1: Compute the elements of M; A, B and C from Ix and Iy.
        a = ix * ix
        b = ix * iy
        c = iy * iy

        # TODO 3.2: Apply the windowing gaussian win_kernel_ on A, B and C.
        a = cv2.sepFilter2D(a, -1, self._win_kernel, self._win_kernel)
        b = cv2.sepFilter2D(b, -1, self._win_kernel, self._win_kernel)
        c = cv2.sepFilter2D(c, -1, self._win_kernel, self._win_kernel)

        # TODO 4: Finish all the corner response functions (see below).
        if self._metric_type == 'harris':
            response = self._harris_metric(a, b, c)
        elif self._metric_type == 'harmonic_mean':
            response = self._harmonic_metric(a, b, c)
        elif self._metric_type == 'min_eigen':
            response = self._min_eigen_metric(a, b, c)
        else:
            raise ValueError("metric_type must be 'harris', 'harmonic_mean' or 'min_eigen'")

        # TODO 5: Dilate image to make each pixel equal to the maximum in the neighborhood.
        local_max = cv2.dilate(response, np.ones((3, 3)))

        # TODO 6: Compute the threshold.
        threshold = response.max() * self._quality_level

        # TODO 7. Extract local maxima above threshold (response > threshold and response == local_max).
        is_strong_and_local_max = (response > threshold) & (response == local_max)
        max_locations = np.transpose(np.nonzero(is_strong_and_local_max))

        keypoint_size = 3.0 * self._window_sigma
        keypoints = np.array([cv2.KeyPoint(float(col), float(row), keypoint_size, -1, response[row, col]) for row, col in max_locations])

        if self._visualize:
            cv2.imshow("Gradient Ix", ix/25)
            cv2.imshow("Gradient Iy", iy/25)
            cv2.imshow("Gradient magnitude", (np.abs(ix) + np.abs(iy))/25)
            cv2.imshow("Image A", a)
            cv2.imshow("Image B", b)
            cv2.imshow("Image C", c)
            cv2.imshow("Response", response / (0.01 * response.max()))
            cv2.imshow("Local max", is_strong_and_local_max.astype(np.uint8) * 255)

        return keypoints, np.asarray(max_locations)

    @staticmethod
    def _harris_metric(a, b, c):
        # TODO 4.1: Finish the Harris metric.
        # Compute the Harris metric for each pixel.
        alpha = 0.06
        det_m = a * c - b * b
        trc_m = a + c

        return det_m - alpha * trc_m * trc_m

    @staticmethod
    def _harmonic_metric(a, b, c):
        # TODO 4.2 Finish the Harmonic Mean metric
        # Compute the Harmonic Mean metric for each pixel.
        det_m = a * c - b * b
        trc_m = a + c

        return det_m * 1./trc_m

    @staticmethod
    def _min_eigen_metric(a, b, c):
        # TODO 4.3 Finish the minimum eigenvalue metric
        # Compute the Min. Eigen metric for each pixel.
        root = np.sqrt(4. * np.square(b) + np.square(a-c))

        return 0.5 * ((a+c) - root)


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
        best_circle = None
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
            # TODO 8: Remove break and perform the correct test!
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

    @staticmethod
    def _least_squares_estimator(points):
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


if __name__ == "__main__":
    run_corners_solution()
