import sys
import os
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from fga import smooth


os.makedirs('../output', exist_ok=True)

img = cv2.imread('../images/engel_sm.bmp', cv2.IMREAD_COLOR)
kernel_size = 7
runs_number = 1

output = smooth(img, kernel_size, n=runs_number)
cv2.imwrite('../output/engel_sm.bmp', output)
