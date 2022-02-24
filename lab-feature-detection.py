import cv2
import numpy as np
import timeit
from corner_detector import CornerDetector
from circle_estimator import CircleEstimator, extract_inlier_points


def retain_best(keypoints, num_to_keep):
    num_to_keep = np.minimum(num_to_keep, len(keypoints))
    best = np.argpartition([p.response for p in keypoints], -num_to_keep)[-num_to_keep:]
    return best


def draw_corner_result(img, keypoints, duration):
    cv2.putText(img, f"corner time: {duration:.2f}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0))
    cv2.drawKeypoints(img, keypoints, img, (0, 255, 0))


def draw_circle_result(img, keypoints, circle_estimate, duration):
    # If not a result
    if not circle_estimate:
        return

    cv2.putText(img, f"circle time: {duration:.2f}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0))

    # Extract and draw circle point inliers
    inlier_pts = extract_inlier_points(circle_estimate, keypoints)
    cv2.drawKeypoints(img, inlier_pts, img, (0, 0, 255))

    # Draw estimated circle
    center = np.asarray(circle_estimate.circle.center, dtype=int)
    radius = int(circle_estimate.circle.radius)
    cv2.circle(img, np.flip(center), radius, (0, 0, 255), cv2.LINE_4, cv2.LINE_AA)


def run_detection_solution():
    # Connect to the camera.
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source {video_source}")
        return
    else:
        print(f"Successfully opened video source {video_source}")

    # Create window
    window_name = 'Window name'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Construct the corner detector.
    # Play around with the parameters!
    # When the second argument is true, additional debug visualizations are shown.
    det = CornerDetector(metric_type='harris', visualize=True)

    # Construct the circle estimator
    estimator = CircleEstimator()

    while True:
        # Read next frame.
        success, frame = cap.read()
        if not success:
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


if __name__ == "__main__":
    run_detection_solution()
