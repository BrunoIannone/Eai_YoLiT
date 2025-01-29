import torch
# from autodistill_grounded_sam_2 import GroundedSAM2
# from autodistill.detection import CaptionOntology
# from autodistill_yolov8 import YOLOv8
from torchvision.io import read_image

import utils
import albumentations as A
import cv2 as cv
import numpy as np
dir = "/home/bruno/Scaricati/Dataset/GazeCapture"  # 03454
sample = "03454/"
# Replace with the path to your JSON file
f_json_file = dir + sample + 'appleFace.json'
l_json_file = dir + sample + "appleLeftEye.json"
r_json_file = dir + sample + "appleRightEye.json"

images_dir = dir + sample + 'frames'       # Directory containing input images

utils.draw_bounding_boxes_from_list(utils.get_bbox_from_txt(dir + "/labels" + "/train" + "/03454_0.txt"),dir+"/images"+"/train"+"/03454_0.jpg")