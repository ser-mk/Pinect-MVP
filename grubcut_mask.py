#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 21:49:05 2017

@author: ?
"""
# Python 2/3 compatibility


import numpy as np
import cv2
import sys

#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.

This sample shows interactive image segmentation using grabcut algorithm.

USAGE:
    python grabcut.py <filename>

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '4' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''



BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}


class GrubCutMask:

    suffix_file = ".grubcut.txt"

    def __init__(self, filename = None, image = None):
        # setting up flags
        self.rect = (0,0,1,1)
        self.drawing = False         # flag for drawing curves
        self.rectangle = False       # flag for drawing rect
        self.rect_over = False       # flag to check if rect drawn
        self.rect_or_mask = 100      # flag for selecting rect or mask mode
        self.value = DRAW_FG         # drawing initialized to FG
        self.thickness = 3           # brush thickness
        self.reset = True
        if image is None :
            self.img = cv2.imread(filename)
        else:
            self.img = image
        self.filename = filename
        self.img2 = self.img.copy()      # a copy of original image
        self.ix = 0
        self.iy = 0
        self.mask = np.zeros(self.img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
        self.name_window = 'grubcut'

    def loadMask(self):
        try :
            mask = np.loadtxt(self.filename + self.suffix_file,dtype=np.uint8)
            if not mask.shape == self.img.shape[:2]:
                print("mask shape is not equal image shape")
                print("mask : ", mask.shape)
                print("image : ", self.img.shape)
            mask = np.where(mask==255,1,0).astype('uint8')
            self.mask = mask
            print("load mask from file")
        except Exception as inst:
            print("don't find mask file ", self.filename,  self.suffix_file)
            print(type(inst))     # the exception instance
            print(inst.args)      # arguments stored in .args
            print(inst)           # __str__ allows args to be printed directly

    def onmouse(self,event,x,y,flags,param):

        if self.reset :
            # Draw Rectangle
            if event == cv2.EVENT_RBUTTONDOWN:
                self.rectangle = True
                self.ix, self.iy = x,y

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.rectangle == True:
                    self.img = self.img2.copy()
                    cv2.rectangle(self.img,(self.ix,self.iy),(x,y),BLUE,2)
                    rect = (min(self.ix,x),min(self.iy,y),abs(self.ix-x),abs(self.iy-y))
                    self.rect_or_mask = 0

            elif event == cv2.EVENT_RBUTTONUP:
                self.rectangle = False
                self.rect_over = True
                cv2.rectangle(self.img,(self.ix,self.iy),(x,y),BLUE,2)
                self.rect = (min(self.ix,x),min(self.iy,y),abs(self.ix-x),abs(self.iy-y))
                self.rect_or_mask = 0
                print(" Now press the key 'n' a few times until no further change \n")
                self.reset = False

        # draw touchup curves
        self.thickness = cv2.getTrackbarPos('thickness', self.name_window)

        if event == cv2.EVENT_LBUTTONDOWN :
            if self.rect_over == False:
                print("first draw rectangle \n")
            else:
                self.drawing = True
                cv2.circle(self.img,(x,y),self.thickness,self.value['color'],-1)
                cv2.circle(self.mask,(x,y),self.thickness,self.value['val'],-1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.circle(self.img,(x,y),self.thickness,self.value['color'],-1)
                cv2.circle(self.mask,(x,y),self.thickness,self.value['val'],-1)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv2.circle(self.img,(x,y),self.thickness,self.value['color'],-1)
                cv2.circle(self.mask,(x,y),self.thickness,self.value['val'],-1)


    def grubcut(self) :
        self.loadMask()

        output = np.zeros(self.img.shape,np.uint8)           # output image to be shown

        def nothing(*arg):
            pass
    
        cv2.namedWindow(self.name_window, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.name_window,self.onmouse)
        cv2.createTrackbar( 'thickness', self.name_window, 14, 22, nothing)

        print(" Instructions: \n")
        print(" Draw a rectangle around the object using right mouse button \n")

        while(1):

            cv2.imshow(self.name_window,output)
            k = cv2.waitKey(1) #- 0x100000

            # key bindings
            if k == 27 or k == ord(' '):         # esc to exit
                break
            elif k == ord('4'): # BG drawing
                print(" mark background regions with left mouse button \n")
                self.value = DRAW_BG
            elif k == ord('1'): # FG drawing
                print(" mark foreground regions with left mouse button \n")
                self.value = DRAW_FG
            elif k == ord('2'): # PR_BG drawing
                self.value = DRAW_PR_BG
            elif k == ord('3'): # PR_FG drawing
                self.value = DRAW_PR_FG
            elif k == ord('s'): # save mask
                if not self.filename == None :
                    np.savetxt(self.filename + self.suffix_file, mask_output, fmt='%u')
                    print(" Result mask save to txt file\n")
                else:
                    print("filename is empty")
            elif k == ord('r'): # reset everything
                print("resetting \n")
                self.rect = (0,0,1,1)
                self.drawing = False
                self.rectangle = False
                self.rect_or_mask = 100
                self.rect_over = False
                self.value = DRAW_FG
                self.img = self.img2.copy()
                self.reset = True
                self.mask = np.zeros(self.img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
                output = np.zeros(self.img.shape,np.uint8)           # output image to be shown
            elif k == ord('n'): # segment the image
                print(""" For finer touchups, mark foreground and background after pressing keys 1-4
                and again press 'n' \n""")
                if (self.rect_or_mask == 0):         # grabcut with rect
                    bgdmodel = np.zeros((1,65),np.float64)
                    fgdmodel = np.zeros((1,65),np.float64)
                    cv2.grabCut(self.img2,self.mask,self.rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
                    self.rect_or_mask = 1
                elif self.rect_or_mask == 1:         # grabcut with mask
                    bgdmodel = np.zeros((1,65),np.float64)
                    fgdmodel = np.zeros((1,65),np.float64)
                    cv2.grabCut(self.img2,self.mask,self.rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)

            mask_output = np.where((self.mask==1) + (self.mask==3),255,0).astype('uint8')
            output = cv2.bitwise_and(self.img2,self.img2,mask=mask_output)
            output = np.hstack((self.img,output))

        cv2.destroyWindow(self.name_window)

        return mask_output



if __name__ == '__main__':

    # print documentation
    print(__doc__)

    # Loading images
    if len(sys.argv) == 2:
        filename = sys.argv[1] # for drawing purposes
    else:
        print("No input image given, so loading default image, ../data/lena.jpg \n")
        print("Correct Usage: python grabcut.py <filename> \n")
        filename = '/home/ser/PIPI/led_line/filter/IPC_2017-07-08.17.43.06.6960.jpg'

    img = cv2.imread(filename)

    GC = GrubCutMask(filename, img)

    output_mask = GC.grubcut()

    cv2.imshow('output_mask' , output_mask)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
