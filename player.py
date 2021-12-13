#!/usr/bin/env python2

import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from calc_mat import Processing
from rectangle_laser import getRectLaser, LaserMarked


class Options:
    lm = LaserMarked()
    start_num_frame = 0
    init = False


if __name__ == '__main__':

    # filename = './led_line/150R.2017-07-02.18.18.55.mp4'
    # filename = './led_line/ccd/IPS_2017-08-27.19.15.08.mp4'
    ##filename = './led_line/ccd0309/1.mp4'
    ##filename = './led_line/ccd0309/IPS_2017-09-03.15.03.55.mp4'
    ##filename = './led_line/ccd0309/IPS_2017-09-03.15.07.27.mp4'
    ##filmename = './led_line/ccd0309/IPS_2017-09-03.15.03.55.mp4'
    # filename = './led_line/ccd0309/IPS_2017-09-03.16.01.34.mp4'
    # filename = './led_line/ccd0309/IPS_2017-09-03.16.04.13.mp4'
    # filename = './led_line/ccd0309/IPS_2017-09-03.16.54.40.mp4'
    # filename = './led_line/ccd0309/IPS_2017-09-03.16.43.17.mp4'
    # filename = './led_line/ccd0309/IPS_2017-09-03.16.46.13.mp4'

    filename = sys.argv[1]

    suffix_opt_file = '.opt'

    cap = cv2.VideoCapture(filename)
    prev_mask = None
    current_mask = prev_mask
    not_loop_frame = False
    laser_calc_proccesing = False


    def nothing(*arg):
        pass


    common_window = 'player'
    speed_bar = 'speed'

    cv2.namedWindow(common_window)
    cv2.createTrackbar(speed_bar, common_window, 20, 1888, nothing)


    class terminateFrameLoop(Exception):
        pass  # declare a label


    opt = Options()
    opt.start_num_frame = 0
    num_frame = 0;

    proc = None
    key = None
    pos = None
    try:
        while (True):
            # Capture frame-by-frame
            ret, next_frame = cap.read()

            num_frame += 1

            if not ret:
                print("end of file! all frames:", num_frame)
                break

            while num_frame < opt.start_num_frame:
                num_frame += 1
                cap.grab()

            # cicrle proccesing frame if need
            while True:

                frame = next_frame.copy()

                example_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                example_bgr = frame.copy()

                if opt.lm.mask is not None:
                    color_mask = cv2.cvtColor(opt.lm.mask, cv2.COLOR_GRAY2BGR)
                    color = np.zeros(color_mask.shape, np.uint8)
                    color[:] = (0, 0, 255)
                    color_mask = cv2.bitwise_and(color_mask, color)
                    cv2.rectangle(color_mask, (opt.lm.x1, opt.lm.y1), (opt.lm.x2, opt.lm.y2), (0, 255, 0), 1)
                    dst = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
                    if pos is not None:
                        print("pos: ", pos)
                        cv2.line(dst, (pos, 0), (pos, dst.shape[0]), (222, 0, 0), 3)
                else:
                    dst = frame

                cv2.putText(dst, str(num_frame), (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

                if laser_calc_proccesing and not_loop_frame:
                    pos = proc.step(frame, key)

                # Display the resulting frame
                cv2.imshow(common_window, dst)

                period = cv2.getTrackbarPos(speed_bar, common_window) + 5

                key = cv2.waitKey(period)

                if key == ord('q'):
                    print("your press 'q' key and exit execute =)")
                    raise terminateFrameLoop()

                if key == ord(' '):
                    print("pause ", not_loop_frame)
                    not_loop_frame = not not_loop_frame

                if key == ord('x'):
                    if proc is None:
                        proc = Processing(opt.lm)
                    laser_calc_proccesing = not laser_calc_proccesing
                    print("stream analyze", "enable" if laser_calc_proccesing else "disable",
                          f" press 'x' to switch this option")

                if key == ord('m'):
                    print("laser marked procces...")
                    opt.lm = getRectLaser(filename, frame)
                    proc = Processing(opt.lm)
                    laser_calc_proccesing = True

                if key == ord('l'):
                    try:
                        print('load option from ', filename, suffix_opt_file)
                        opt = pickle.load(open(filename + suffix_opt_file, "rb"))
                    except Exception as inst:
                        print("don't find options file ", filename, suffix_opt_file)
                        print(type(inst))  # the exception instance
                        print(inst.args)  # arguments stored in .args
                        print(inst)  # __str__ allows args to be printed directly

                if key == ord('s'):
                    opt.start_num_frame = num_frame
                    try:
                        print('save option in ', filename, suffix_opt_file)
                        pickle.dump(opt, open(filename + suffix_opt_file, "wb"))
                    except Exception as inst:
                        print("don't find options file ", filename, suffix_opt_file)
                        print(type(inst))  # the exception instance
                        print(inst.args)  # arguments stored in .args
                        print(inst)  # __str__ allows args to be printed directly

                if not_loop_frame:
                    break

    except terminateFrameLoop:
        print("terminate laser proccesing")
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    plt.close()
