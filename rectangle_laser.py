#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 00:28:52 2017

@author: ser
"""
import cv2
import sys
from  grubcut_mask import GrubCutMask

class LaserMarked:
    mask = None
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0

def getRectLaser(filename = None, image = None):
        
    GC = GrubCutMask(filename, image)

    output_mask = GC.grubcut()
    
    contour_laser = output_mask.copy()
    
    im2, contours, hierarchy = cv2.findContours(contour_laser,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    max_counter = max(contours, key=cv2.contourArea)
    
    x,y,w,h = cv2.boundingRect(max_counter)
    
    ret = LaserMarked()
    ret.mask = output_mask
    ret.x1 = x
    ret.x2 = x + w
    ret.y1 = y
    ret.y2 = y + h
    
    return ret
    

if __name__ == '__main__':

    # Loading images
    if len(sys.argv) == 2:
        filename = sys.argv[1] # for drawing purposes
    else:
        filename = './led_line/filter/IPC_2017-07-08.17.43.06.6960.jpg'

    img = cv2.imread(filename)

    lm = getRectLaser(filename, img)
    
    color_mask = cv2.cvtColor(lm.mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color_mask,(lm.x1,lm.y1),(lm.x2, lm.y2),(0,255,0),1)
    
    dst = cv2.addWeighted(img,0.7, color_mask,0.3, 0)
    
    cv2.namedWindow('output_mask')
    
    cv2.imshow('output_mask' , dst)
    cv2.waitKey(0)


    cv2.destroyAllWindows()
    