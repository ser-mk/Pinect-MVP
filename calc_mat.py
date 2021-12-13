#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 23:37:05 2017

@author: ser
"""

import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from rectangle_laser import getRectLaser

_MAX_64 = 18446744073709551615


class Processing:
    def __init__(self, lm):
        self.lm = lm
        plt.ion()
        self.fig = plt.figure()
        self.count = 0
        self.SIZE_X_HIST = (self.lm.x2 - self.lm.x1)
        self.min_hist = np.zeros(self.SIZE_X_HIST, dtype=np.uint64)
        self.min_hist.fill(_MAX_64)
        self.max_hist = np.zeros(self.SIZE_X_HIST, dtype=np.uint64)
        self.study_min = True
        self.min_power = _MAX_64
        self.min_width_power = _MAX_64
        self.study_power = False

    def _sumPointsBW(self, bw_image):

        diffx = self.lm.x2 - self.lm.x1
        diffy = self.lm.y2 - self.lm.y1

        vertical = True

        if diffy > diffx:
            vertical = False

        if not vertical:
            pass

        dst = cv2.bitwise_and(self.lm.mask, bw_image)
        region = dst[self.lm.y1:self.lm.y2, self.lm.x1:self.lm.x2]

        vert_hist = region.sum(axis=0)

        return vert_hist

    def _sumPointsRGB(self, rgb_image):

        diffx = self.lm.x2 - self.lm.x1
        diffy = self.lm.y2 - self.lm.y1

        vertical = True

        if diffy > diffx:
            vertical = False

        if not vertical:
            pass

        dst = cv2.bitwise_and(rgb_image, rgb_image, mask=self.lm.mask)
        region = dst[self.lm.y1:self.lm.y2, self.lm.x1:self.lm.x2]
        vert_hist = region.sum(axis=0)
        vert_hist = vert_hist.sum(axis=1)

        return vert_hist

    def _checkMinEl(self, hist):
        if self.min_hist.shape != hist.shape:
            print("self.min_hist.shape != hist.shape");
            return
        for i, x in enumerate(self.min_hist):
            if x > hist[i]:
                self.min_hist[i] = hist[i]

    def _decMinHist(self, koeff):
        print("decrease min his on 1/", koeff)
        self.min_hist -= self.min_hist // koeff

    def _substractHist(self, hist):
        sub = np.zeros(self.SIZE_X_HIST, dtype=np.uint64)

        for i, x in enumerate(self.min_hist):
            if x <= hist[i]:
                continue
            sub[i] = x - hist[i]
        return sub

    def _calcPower(self, sub):
        power = np.zeros(self.SIZE_X_HIST, dtype=np.uint64)
        for i, x in enumerate(sub):
            if i == 0:
                continue
            if x == 0:
                continue

            power[i] = power[i - 1] + x

        return power

    def _minPower(self, power):
        max_power = 0
        width_max_power = 0
        for i, x in enumerate(power):
            if i == 0:
                continue
            prev_power = power[i - 1]
            if (x == 0) and (prev_power > self.min_hist[i - 1]):
                if prev_power <= max_power:
                    continue
                j = i - 1
                while j > 0:
                    if power[j] == 0:
                        break
                    j -= 1
                width_max_power = i - j
                max_power = prev_power
                continue
        if max_power >= self.min_power:
            return max_power, width_max_power

        if max_power == 0:
            return max_power, width_max_power

            # TODO bound width
        self.min_power = max_power
        self.min_width_power = width_max_power

        return self.min_power, self.min_width_power

    def _findMaxPower(self, power):
        max_power = 0
        width_max_power = 0
        pos = 0
        for i, x in enumerate(power):
            if i == 0:
                continue
            prev_power = power[i - 1]
            if (x == 0) and (prev_power > self.min_hist[i - 1]):
                if prev_power <= max_power:
                    continue
                j = i - 1
                while j > 0:
                    if power[j] == 0:
                        break
                    j -= 1
                width_max_power = i - j
                max_power = prev_power
                pos = j + width_max_power / 2
                continue
        if max_power == 0:
            return None

        return int(pos + self.lm.x1)

    def step(self, image, key):

        if self.lm.mask.shape != image.shape[:2]:
            print("shape mask and image not equal");
            return

        if self.lm == None:
            print("laser marker ot exist")
            return

        _sumPoint = None
        if len(image.shape) == 2:
            _sumPoint = self._sumPointsBW
        else:
            _sumPoint = self._sumPointsRGB

        self.count += 1
        vert_hist = _sumPoint(image)

        plt.clf()
        plt.title('Ultra-narrow 1D image convolution. Cadr:' + str(self.count))

        if key == ord('t'):
            print("study min", self.study_min)
            self.study_min = not self.study_min

        if key == ord('p'):
            print("study power min", self.study_power)
            self.study_power = not self.study_power

        if key == ord('d'):
            self._decMinHist(20)

        pos = None

        if self.study_min:
            self._checkMinEl(vert_hist)
        else:
            sub = self._substractHist(vert_hist)
            power = self._calcPower(sub)
            plt.plot(sub)
            plt.plot(power)
            if self.study_power:
                power, width = self._minPower(power)
                sup = "p: " + str(power) + " w: " + str(width) + " gp: " + \
                      str(self.min_power) + " gw: " + str(self.min_width_power)
                self.fig.suptitle(sup)
            else:
                pos = self._findMaxPower(power)

        plt.plot(self.min_hist)

        plt.plot(vert_hist)
        self.fig.canvas.draw()

        time.sleep(1e-6)  # unnecessary, but useful

        return pos


if __name__ == '__main__':

    import random

    # Loading images
    if len(sys.argv) == 2:
        filename = sys.argv[1]  # for drawing purposes
    else:
        filename = 'res/capture_1.jpg'

    img = cv2.imread(filename)

    lm = getRectLaser(filename, img)

    color_mask = cv2.cvtColor(lm.mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color_mask, (lm.x1, lm.y1), (lm.x2, lm.y2), (0, 255, 0), 1)

    dst = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    cv2.namedWindow('output_mask')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp = img.copy()

    pp = Processing(lm)
    k = 0

    for i in range(0, 111):
        temp = img + cv2.randn(temp, (0), (11))
        if i > 20:
            temp[:, 50:70, :] = random.randint(40, 90)

        pos = pp.step(temp, k)

        print("pos: ", pos)

        cv2.imshow('output_mask', temp)
        k = cv2.waitKey(0)

        if k == ord('q'):
            break

    plt.close()
    cv2.destroyAllWindows()
