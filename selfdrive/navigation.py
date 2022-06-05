import statistics
import math
import cv2
import numpy as np
from collections import namedtuple
from typing import List

Range = namedtuple("Range", ["lower", "upper"])
Point = namedtuple("Point", ["x", "y"])
Line = namedtuple("Line", ["pt1", "pt2", "slope"])
Rectangle = namedtuple("Rectangle", ["pt1", "pt2"])
NavScores = namedtuple("NavScores", ["forward", "right_count", "right_score", "left_count", "left_score"])

RIGHT_MAX_VALUE = 350
LEFT_MAX_VALUE = 800


class Masks:
    yellow = Range(lower=[22, 93, 0], upper=[30, 255, 255])


class Directions:
    right = 0
    left = 1


def get_turn_detection_region(width, height) -> List:
    return [
        (0, height // 2),
        (width, height // 2),
        (width, height),
        (0, height)
    ]


def get_steer_adjust_region(width, height) -> List:
    return [
        (0, 0),
        (width, 0),
        (width, height),
        (0, height)
    ]


def get_non_zero_rectangle(width, height, direction) -> Rectangle:
    if direction == Directions.right:
        return Rectangle(
            pt1=Point(x=int(2 * width / 3), y=int(height / 2.5)),
            pt2=Point(x=width, y=int(4 * height / 5))
        )


def distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def mask_img(img, mask: Range):
    original = img.copy()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(mask.lower, dtype="uint8")
    upper = np.array(mask.upper, dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    masked = cv2.bitwise_and(original, original, mask=mask)
    return masked


def crop_region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def detect_lanes(img, mask: Range, vertices):
    masked = mask_img(img, mask)
    canny_img = cv2.Canny(masked, 100, 200)
    cropped_image = crop_region_of_interest(
        img=canny_img,
        vertices=np.array([vertices], np.int32)
    )
    lines = cv2.HoughLinesP(cropped_image,
                            rho=6,
                            theta=np.pi / 180,
                            threshold=160,
                            lines=np.array([]),
                            minLineLength=20,
                            maxLineGap=100)
    return lines, masked


def filter_lines(lines, min_slope=None):
    if lines is None:
        return None
    filtered_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x1 - x2) > 25:
                m = (y2 - y1) / (x2 - x1)
                if min_slope is None or abs(m) > min_slope:
                    filtered_lines.append(Line(
                        pt1=Point(x1, y1),
                        pt2=Point(x2, y2),
                        slope=m
                    ))
    ret_mean_slope = 0
    if len(filtered_lines) > 0:
        mean_slope = statistics.mean([l.slope for l in filtered_lines])
        for line in filtered_lines:
            if abs(line.slope - mean_slope) > 0.1:
                filtered_lines.remove(line)
        ret_mean_slope = statistics.mean([l.slope for l in filtered_lines])

    return filtered_lines, ret_mean_slope


def get_non_zero_pixels(masked, rectangle: Rectangle):
    pt1, pt2 = rectangle
    masked = masked[pt1.y:pt2.y, pt1.x:pt2.x]
    masked = cv2.cvtColor(masked, cv2.COLOR_RGBA2GRAY)
    ret, thresh = cv2.threshold(masked, 127, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thresh)


def process_lines(image, filtered_lines: List[Line]):
    height, width = image.shape[:2]
    rec_left_y = height - 60
    rec_right_x = int(width / 10)
    rec_right_y = height
    forward = 0
    right_count, right_score = 0, 0
    left_count, left_score = 0, 0
    right_x_cords = []
    left_y_cords = []

    max_y_cord = -1
    for line in filtered_lines:
        count_before = forward
        for i in range(2):
            x, y = line[i]
            if x < rec_right_x and rec_left_y < y < rec_right_y:
                forward += 1
            if y > max_y_cord:
                max_y_cord = y
        if count_before == forward:
            # No forward lines -> right / left
            for i in range(2):
                x, y = line[i]
                if x < rec_right_x and y < rec_left_y:
                    left_count += 1
                    left_y_cords.append(y)
                    break
                if x > rec_right_x and y > rec_left_y:
                    right_count += 1
                    right_x_cords.append(x)
                    break

    if max_y_cord < 400:
        return None

    if len(right_x_cords) > 0 and forward < 5:
        mean_right_point = Point(x=int(statistics.mean(right_x_cords)), y=rec_left_y)
        line_slope = (mean_right_point.y - height) / (mean_right_point.x - 0)
        right_score = (0.8 - abs(line_slope)) * distance(mean_right_point, (0, height)) / RIGHT_MAX_VALUE

    if len(left_y_cords) > 0 and forward < 5:
        mean_left_point = Point(x=40, y=int(statistics.mean(left_y_cords)))
        line_slope = (mean_left_point.y - height) / (mean_left_point.x - 0)
        left_score = abs(line_slope) * distance(mean_left_point, (0, height)) / LEFT_MAX_VALUE

    return NavScores(
        forward=forward,
        right_count=right_count,
        right_score=right_score,
        left_count=left_count,
        left_score=left_score
    )
