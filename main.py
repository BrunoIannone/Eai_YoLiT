import torch
# from autodistill_grounded_sam_2 import GroundedSAM2
# from autodistill.detection import CaptionOntology
# from autodistill_yolov8 import YOLOv8

import utils


dir = "/home/bruno/Scaricati/Dataset/GazeCapture/01734/"  # 03454
f_json_file = dir + 'appleFace.json'  # Replace with the path to your JSON file
l_json_file = dir + "appleLeftEye.json"
r_json_file = dir + "appleRightEye.json"

images_dir = dir + 'frames'       # Directory containing input images

utils.draw_bounding_boxes_opencv(
    f_json_file, l_json_file, r_json_file, images_dir, True)


# cv.imshow("Pisellone",cv.imread("/home/bruno/Scaricati/Dataset/GazeCapture/00507/frames/00862.jpg"))
# k = cv.waitKey(0) # Wait for a keystroke in the window
