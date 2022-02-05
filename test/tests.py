from unittest import TestCase
import unittest
import cv2
import sys
import os
import cv2


class TestSmoothing(TestCase):
    def test_smoothing_7_1(self):
        """
        Test smoothing results
        """
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.dirname(SCRIPT_DIR))
        from fga import smooth
        
        img = cv2.imread('../images/engel_sm.bmp', cv2.IMREAD_COLOR)
        kernel_size = 7
        runs_number = 1

        output = smooth(img, kernel_size, n=runs_number)
        gt_output = cv2.imread('../images/engel_sm_7_1.bmp', cv2.IMREAD_COLOR)
        
        self.assertTrue((output == gt_output).all())

    def test_smoothing_3_2(self):
        """
        Test smoothing results
        """
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.dirname(SCRIPT_DIR))
        from fga import smooth

        img = cv2.imread('../images/engel_sm.bmp', cv2.IMREAD_COLOR)
        kernel_size = 3
        runs_number = 2

        output = smooth(img, kernel_size, n=runs_number)
        gt_output = cv2.imread('../images/engel_sm_3_2.bmp', cv2.IMREAD_COLOR)

        self.assertTrue((output == gt_output).all())


if __name__ == '__main__':
    tc = TestSmoothing()
    unittest.main()
