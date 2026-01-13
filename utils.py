# utils.py
import cv2
import numpy as np

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, ksize=5):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def canny(img, low=50, high=150):
    return cv2.Canny(img, low, high)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=(0, 255, 0), thickness=10):
    if lines is None:
        return
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.0):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def get_trapezoid_vertices(img_shape, bottom_trim=0.05, top_width=0.4, bottom_width=0.9, top_y=0.6):
    h, w = img_shape[:2]
    bottom_left = (int(w * (1 - bottom_width) / 2), int(h * (1 - bottom_trim)))
    bottom_right = (int(w * (1 + bottom_width) / 2), int(h * (1 - bottom_trim)))
    top_left = (int(w * (0.5 - top_width / 2)), int(h * top_y))
    top_right = (int(w * (0.5 + top_width / 2)), int(h * top_y))
    return np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
