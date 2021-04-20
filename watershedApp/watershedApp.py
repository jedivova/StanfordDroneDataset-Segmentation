#!/usr/bin/env python

'''
Slight modification of https://github.com/opencv/opencv/blob/master/samples/python/watershed.py

Watershed segmentation
=========

This program demonstrates the watershed segmentation algorithm
in OpenCV: watershed().

Usage
-----
watershed.py [image filename]

Keys
----
  1-7   - switch marker color
  SPACE - update segmentation
  r     - reset
  a     - toggle autoupdate
  ESC   - exit

'''


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from pathlib import Path


class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()


class App:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(str(img_path))
        if self.img is None:
            raise Exception('Failed to load image file: %s' % str(img_path))
        h, w = self.img.shape[:2]
        self.markers = np.zeros((h, w), np.int32)
        self.markers_vis = self.img.copy()
        self.cur_marker = 1
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255

        self.auto_update = True
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.moveWindow('img',300,200)
        self.sketch = Sketcher('img', [self.markers_vis, self.markers], self.get_colors)
        self.returnVar = self.markers.copy()

    def get_colors(self):
        '''

        :return:
        '''
        return list(map(int, self.colors[self.cur_marker])), self.cur_marker

    def watershed(self):
        m = self.markers.copy()
        cv2.watershed(self.img, m)
        self.returnVar = m.copy()
        overlay = self.colors[np.maximum(m, 0)]
        vis = cv2.addWeighted(self.img, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
        cv2.namedWindow('watershed', cv2.WINDOW_NORMAL)
        # cv2.moveWindow('watershed',780,200)
        cv2.imshow('watershed', vis)

    def run(self):
        while True:
            ch = 0xFF & cv2.waitKey(50)
            if ch == 27:
                break
            if ch >= ord('1') and ch <= ord('7'):
                self.cur_marker = ch - ord('0')
                print('marker: ', self.cur_marker)
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.watershed()
                self.sketch.dirty = False
            if ch in [ord('a'), ord('A')]:
                self.auto_update = not self.auto_update
                print('auto_update if', ['off', 'on'][self.auto_update])
            if ch in [ord('r'), ord('R')]:
                self.markers[:] = 0
                self.markers_vis[:] = self.img
                self.sketch.show()
            if ch in [ord('g'), ord('G')]:
                markers_p = path.parent.joinpath('markers.png')
                cv2.imwrite(str(markers_p), self.markers)
                print('saved to test_markers.png')
            if ch in [ord('k'), ord('K')]:
                markers_p = path.parent.joinpath('markers.png')
                cv2.imwrite(str(markers_p), self.markers)
                print('saved to test_markers.png')
                print('next image')
                break
        cv2.destroyAllWindows()
        return self.returnVar
        

if __name__ == '__main__':

    print(''' RED is RESTRICTED
            Green is ROAD
            Yellow is BUILDING'''
          )

    SDD_path = r'Q:\Downloads\stanford_campus_dataset\annotations'
    references = list(Path(SDD_path).glob(r'*\*\reference.jpg'))

    p = str(references[0])

    for path in references:
        if not path.parent.joinpath('markers.png').is_file():
            App(path).run()
            print('destroyed, going next')




