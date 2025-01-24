import torch
# from autodistill_grounded_sam_2 import GroundedSAM2
# from autodistill.detection import CaptionOntology
# from autodistill_yolov8 import YOLOv8

import utils
import albumentations as A
import cv2 as cv
import numpy as np
dir = "/home/bruno/Scaricati/Dataset/GazeCapture/train/"  # 03454
sample = "03454/"
# Replace with the path to your JSON file
f_json_file = dir + sample + 'appleFace.json'
l_json_file = dir + sample + "appleLeftEye.json"
r_json_file = dir + sample + "appleRightEye.json"

images_dir = dir + sample + 'frames'       # Directory containing input images

# utils.draw_bounding_boxes_opencv(
#     f_json_file, l_json_file, r_json_file, images_dir, True)

# utils.sort_samples(dir)
# cv.imshow("IMG",cv.imread("/home/bruno/Scaricati/Dataset/GazeCapture/00507/frames/00862.jpg"))
# k = cv.waitKey(0) # Wait for a keystroke in the window
labels = ["face"]

face_bbox = utils.get_bbox_from_json(f_json_file)

face_bbox = face_bbox[0]
face_bbox = [list(face_bbox.values())]
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

print(images_dir + "/0000")
image = cv.imread(images_dir + "/00000.jpg")
#image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
utils.draw_bounding_boxes_from_list(face_bbox, images_dir, True)
transformed = transform(image=image, bboxes=face_bbox, labels=labels)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
print(transformed_image)
start_point = (int(transformed_bboxes[0][0]), int(transformed_bboxes[0][1]))
end_point = (int(transformed_bboxes[0][0]) + int(transformed_bboxes[0][2]), int(transformed_bboxes[0][1]) + int(transformed_bboxes[0][3]))
print(type(transformed_image))
color = (0, 0, 255)  # Red color in BGR
thickness = 2
#transformed_image =cv.imread(transformed_image)

cv.rectangle(transformed_image, start_point, end_point, color, thickness)
cv.imshow("Sample", transformed_image)
k = cv.waitKey(0)
