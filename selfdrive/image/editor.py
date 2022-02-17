import numpy as np
import cv2
from collections import namedtuple

CIRCLE_CROP = namedtuple("circle", ["minDist", "dp", "param1", "param2", "minRadius", "maxRadius"])


class ImageEditor:
    def __init__(self, path=None, img=None):
        self.img = img
        if path:
            self.img = cv2.imread(path)
        self.circle_crop = CIRCLE_CROP(minDist=0, dp=0, param1=0,
                                       param2=0, minRadius=0, maxRadius=1)

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, img):
        self._img = img

    @staticmethod
    def _adjust_image_box(box, shape):
        left, top, right, bottom = box
        while left < 0:
            left, right = left + 1, right + 1
        while right > shape[1]:
            left, right = left - 1, right - 1
        while top < 0:
            top, bottom = top + 1, bottom + 1
        while bottom > shape[0]:
            top, bottom = top - 1, bottom - 1
        return left, top, right, bottom

    def crop_circle_box(self, image=None):
        image = self.img if image is None else image
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray_img,
                                   cv2.HOUGH_GRADIENT,
                                   minDist=self.circle_crop.minDist,
                                   dp=self.circle_crop.dp,
                                   param1=self.circle_crop.param1,
                                   param2=self.circle_crop.param2,
                                   minRadius=self.circle_crop.minRadius,
                                   maxRadius=self.circle_crop.maxRadius)
        ret = []
        if circles is not None:
            for i in range(len(circles[0])):
                circles = np.uint16(np.around(circles))
                x, y, r = (int(x) for x in circles[0][i])
                left, top, right, bottom = self._adjust_image_box(
                    box=(x - 100, y - 100, x + 100, y + 100),
                    shape=image.shape
                )
                ret.append((image[top:bottom, left:right], (x, y, r)))
        return ret

    def crop_search_areas(self, size, cords, function):
        for cord in cords:
            for processed_image in function(
                    self.img[cord[0]:cord[0] + size[0], cord[1]:cord[1] + size[1]]):
                yield processed_image

    def implement_filter(self, im_filter):
        if im_filter.color:
            self.img = self.img.convert(im_filter.color)

        if im_filter.crop:
            top, bottom, left, right = im_filter.crop
            self.img = self.img[top:bottom, left:right]

        if im_filter.shape:
            self.img = cv2.resize(self.img, dsize=im_filter.shape)

        return self.img