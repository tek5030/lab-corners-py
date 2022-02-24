from filters import *


class CornerDetector:
    """A home brewes corner detector!"""

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

        # TODO 2: Estimate image gradients Ix and Iy using _g_kernel and _dg_kernel.
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

        # TODO 4: Finish all the corner response functions.
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

        # TODO 7. Extract local maxima above threshold.
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
