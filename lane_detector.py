# lane_detector.py
import numpy as np
from collections import deque
import cv2
from utils import to_gray, gaussian_blur, canny, region_of_interest, draw_lines, weighted_img, get_trapezoid_vertices

class LaneDetector:
    def __init__(self, debug=False):
        self.debug = debug
        # store recent left/right fits to smooth outputs
        self.recent_left = deque(maxlen=5)
        self.recent_right = deque(maxlen=5)

    def find_lane_lines_by_polyfit(self, binary_warped):
        """
        binary_warped: single-channel binary image (edges)
        returns: left_fit_line (x coords), right_fit_line (x coords), and y-values used
        """
        # histogram + sliding windows approach for robustness
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:], axis=0)
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # sliding windows parameters
        nwindows = 9
        window_height = int(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # if found > minpix pixels, recenter next window
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # concat indices
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # no lanes found
            return None, None, None

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # fit a second degree poly
        if leftx.size == 0 or rightx.size == 0:
            return None, None, None

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return left_fitx, right_fitx, ploty

    def smooth_lines(self, left_fitx, right_fitx):
        if left_fitx is not None:
            self.recent_left.append(left_fitx)
        if right_fitx is not None:
            self.recent_right.append(right_fitx)

        if len(self.recent_left) > 0:
            left_avg = np.mean(np.array(self.recent_left), axis=0)
        else:
            left_avg = left_fitx

        if len(self.recent_right) > 0:
            right_avg = np.mean(np.array(self.recent_right), axis=0)
        else:
            right_avg = right_fitx

        return left_avg, right_avg

    def draw_lane_overlay(self, original_img, left_fitx, right_fitx, ploty, color=(0, 255, 0), alpha=0.3):
        overlay = np.zeros_like(original_img)
        if left_fitx is None or right_fitx is None:
            return original_img
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(overlay, np.int32([pts]), color)
        result = weighted_img(overlay, original_img, α=1 - alpha, β=alpha, γ=0.0) if False else cv2.addWeighted(original_img, 1.0, overlay, alpha, 0)
        # draw center line
        midx = (left_fitx[-1] + right_fitx[-1]) / 2
        cv2.line(result, (int(midx), int(ploty[-1])), (int(original_img.shape[1] // 2), int(ploty[-1])), (255, 0, 0), 2)
        return result

    def process(self, frame, params):
        # params: dict for thresholds and ROI
        gray = to_gray(frame)
        blur = gaussian_blur(gray, params.get('blur_ksize', 5))
        edges = canny(blur, params.get('canny_low', 50), params.get('canny_high', 150))

        verts = params.get('roi_vertices', get_trapezoid_vertices(frame.shape,
                                                                  bottom_trim=0.05,
                                                                  top_width=0.4,
                                                                  bottom_width=0.9,
                                                                  top_y=0.6))
        masked = region_of_interest(edges, verts)

        # find lane lines via polyfit
        left_fitx, right_fitx, ploty = self.find_lane_lines_by_polyfit(masked)
        if left_fitx is None or right_fitx is None:
            # if fail, return overlay with edges only
            overlay_edges = np.zeros_like(frame)
            overlay_edges[masked != 0] = (0, 255, 0)
            out = cv2.addWeighted(frame, 0.9, overlay_edges, 1.0, 0)
            return out, {'edges': edges, 'masked': masked, 'left': None, 'right': None}

        left_avg, right_avg = self.smooth_lines(left_fitx, right_fitx)
        overlay = self.draw_lane_overlay(frame, left_avg, right_avg, ploty, color=(0, 255, 0), alpha=0.25)
        return overlay, {'edges': edges, 'masked': masked, 'left': left_avg, 'right': right_avg}
