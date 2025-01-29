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

# # utils.draw_bounding_boxes_opencv(
# #     f_json_file, l_json_file, r_json_file, images_dir, True)

# # utils.sort_samples(dir)
# # cv.imshow("IMG",cv.imread("/home/bruno/Scaricati/Dataset/GazeCapture/00507/frames/00862.jpg"))
# # k = cv.waitKey(0) # Wait for a keystroke in the window
# labels = ["face","l_eye"]

# face_bbox = utils.get_bbox_from_json(f_json_file)
# l_bbox = utils.get_bbox_from_json(l_json_file)
# #print(face_bbox)
# face_bbox = utils.bbox_dict_to_list_of_list(face_bbox)
# l_bbox = utils.bbox_dict_to_list_of_list(l_bbox)

# face_bbox = [face_bbox[0],face_bbox[1]]
# print(face_bbox)
# l_bbox = [l_bbox[0],l_bbox[1]]
# print(l_bbox)
# for i, elem in enumerate(l_bbox):
#     l_bbox_elem =l_bbox[i]
#     l_bbox_elem[0] = l_bbox_elem[0] + face_bbox[i][0]
#     l_bbox_elem[1] = l_bbox_elem[1] + face_bbox[i][1]
#     l_bbox[i] = l_bbox_elem
# print(l_bbox)
# # face_bbox = [list(face_bbox.values())]
# transform = A.Compose([
#     A.RandomCrop(width=450, height=450),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
# ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

# # print(images_dir + "/0000")
# image = []
# image.append(cv.imread(images_dir + "/00000.jpg"))
# image.append(cv.imread(images_dir + "/00001.jpg"))

# for i,img in enumerate(image):
#     #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     #cv.imshow("Sample", img)
#     #k = cv.waitKey(0)

# # utils.draw_bounding_boxes_from_list(face_bbox, images_dir, True)
#     transformed = transform(image=img, bboxes=[face_bbox[i],l_bbox[i]], labels=labels)
#     transformed_image = transformed['image']
#     transformed_bboxes = transformed['bboxes']
#     #print(transformed_bboxes)
# # print(transformed_image)
#     start_point = (int(transformed_bboxes[0][0]), int(transformed_bboxes[0][1]))
#     end_point = (int(transformed_bboxes[0][0]) + int(transformed_bboxes[0][2]), int(transformed_bboxes[0][1]) + int(transformed_bboxes[0][3]))
# # print(type(transformed_image))
#     color = (0, 0, 255)  # Red color in BGR
#     thickness = 2
#     #transformed_image =cv.imread(transformed_image)

#     cv.rectangle(transformed_image, start_point, end_point, color, thickness)
    
#     l_start_point = (int(transformed_bboxes[1][0]), int(transformed_bboxes[1][1] ))
#     l_end_point = (int(transformed_bboxes[1][0] ) + int(transformed_bboxes[1][2]), int(transformed_bboxes[1][1]) + int(transformed_bboxes[1][3]))
# # print(type(transformed_image))
#     color = (0, 0, 255)  # Red color in BGR
#     thickness = 2
#     cv.rectangle(transformed_image, l_start_point, l_end_point, color, thickness)

    
    
#     cv.imshow("Sample", transformed_image)
#     k = cv.waitKey(0)
utils.convert_gazecapture_for_yolo(dir)
#utils.count_samples(dir)