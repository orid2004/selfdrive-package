import statistics
import cv2.cv2 as cv2
from .navigation import Rectangle, Line, NavScores
from typing import List


def visualize_non_zero_pixels(img, rectangle: Rectangle, non_zero_pixels):
    cv2.rectangle(img, rectangle.pt1, rectangle.pt2, (0, 255, 0), thickness=3)
    cv2.putText(img, f'CountNonZero: {non_zero_pixels}', (rectangle.pt1.x, rectangle.pt1.y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
                cv2.LINE_AA)
    return img


def visualize_lane_detection(img, filtered_lines: List[Line], scores: NavScores = None):
    height, width = img.shape[:2]
    for line in filtered_lines:
        cv2.line(img, line.pt1, line.pt2, (255, 255, 255), thickness=2)
    cv2.rectangle(img, (0, height - 60), (width // 10, height), (255, 0, 0), thickness=5)
    if scores:
        text = f"Forward: {scores.forward}\nRight: {scores.right_count}\nLeft: {scores.left_count}".split('\n')
        for i in range(len(text)):
            cv2.putText(img, text[i], (25, 50 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 4, cv2.LINE_AA)
    return img


def visualize_lane_detection_side_view(img, filtered_lines: List[Line]):
    for line in filtered_lines:
        cv2.line(img, line.pt1, line.pt2, (255, 255, 255), thickness=2)
    if len(filtered_lines) > 0:
        mean_slope = statistics.mean([i.slope for i in filtered_lines])
    else:
        mean_slope = '?'
    cv2.putText(img, f"Slope: {mean_slope}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 4, cv2.LINE_AA)
    return img
