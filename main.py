import torch
# from autodistill_grounded_sam_2 import GroundedSAM2
# from autodistill.detection import CaptionOntology
# from autodistill_yolov8 import YOLOv8 #TODO: Forse per funzionare bisognerà fare downgrade di ultralytics
#from torchvision.io import read_image
from ultralytics import YOLO
import utils
import albumentations as A
import cv2 as cv
import numpy as np
import os
import argparse
from yolo_params import yolo_params_,dir,checkpoint
def main():
    parser = argparse.ArgumentParser(description="Training main")
    parser.add_argument("--model", type=str, help="model to train can be yolo or",required=True)
    parser.add_argument("--yolo_size", type=str, help="Choose yolo model size, can be n,s,m" ,required=False,default="s")

    parser.add_argument("--mode", type=str, help="can be train or predict" ,required=True)
    parser.add_argument("--sample_id", type=str, help="Sample id to predict" ,required=False)
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path" ,required=False,default= checkpoint)

    
    args = parser.parse_args()

    if args.model == "yolo" and args.mode == "train":
        
        model = YOLO("yolov8" + args.yolo_size + ".pt")
        results = model.train(**yolo_params_)
    elif args.model == "yolo" and args.mode == "predict":
        model = YOLO(args.checkpoint)

        results = model(os.path.join(dir, "/images/test/",args.sample_id,".jpg"))  # predict on an image

        image_with_boxes = results[0].plot()

        # Display the image using OpenCV
        cv.imshow("YOLOv8 Detection", image_with_boxes)
        cv.waitKey(0)
        cv.destroyAllWindows()



if __name__ == "__main__":
    main()
        
# dir = "/home/bruno/Scaricati/Dataset/GazeCapture"  # 03454
# sample = "03454/"
# Replace with the path to your JSON file
# f_json_file = dir + sample + 'appleFace.json'
# l_json_file = dir + sample + "appleLeftEye.json"
# r_json_file = dir + sample + "appleRightEye.json"

# images_dir = dir + sample + 'frames'       # Directory containing input images

# utils.draw_bounding_boxes_from_list(utils.get_bbox_from_txt(dir + "/labels" + "/train" + "/03454_0.txt"),dir+"/images"+"/train"+"/03454_0.jpg")
# model = YOLO("yolov8s.pt")
# model = YOLO("/home/bruno/Desktop/Eai_YoLiT/runs/detect/train7/weights/best.pt")

# yolo_params = {
#     "data": dir + "/data.yaml", 
#    "fraction":1.0, #Allows for training on a subset of the full dataset
#     "epochs": 5, 
#     "imgsz": 640,
#     "batch":-1,
#     "lr0":1e-3,
#     "patience":2,
#     "save_period":-1,
#     "workers": 2,
#     "optimizer":"AdamW",
#     "freeze":10,
#     "dropout":0.2,
#     "warmup_epochs":2,
#     "plots": True,
#     #AUGMENTATION##
#     "degrees":90.0,
#     "perspective":0.0001,
#     "fliplr":0.0,
#     "mosaic":0.0,

#     }

# results = model.train(**yolo_params)

# import os

# Set your labels folder path

#utils.convert_gazecapture_for_yolo(dir,["face","l_eye","r_eye"])
#print("Processing complete!")
#utils.draw_yolo_bboxes(dir + "/images/test/00010_0.jpg",utils.get_bbox_from_txt(dir + "/labels/test/00010_0.txt"),["face","l_eye","r_eye"])
#model = YOLO("yolov8n.pt")  # load a custom model

# Predict with the model
#metrics = model.val(data = dir + "/data.yaml",batch = 64)
# results = model(dir + "/images/test/00010_1420.jpg")  # predict on an image

# image_with_boxes = results[0].plot()

# # Display the image using OpenCV
# cv.imshow("YOLOv8 Detection", image_with_boxes)
# cv.waitKey(0)
# cv.destroyAllWindows()