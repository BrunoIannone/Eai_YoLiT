
to download the dataset:
wget -O gazecapture.tar "https://gazecapture.csail.mit.edu/dataset.php?email=iannone.1883472@studenti.uniroma1.it&hash=e744f91c29ec99f0e662c9177946c627"
to install distil yolo: pip3 install autodistill-grounded-sam-2 autodistill-yolov8 ( se dà un errore su cuda_home, dare sudo apt install nvidia-cuda-toolkit)


per recuperare tutte le bounding box valide di un sample: 

   
dir = "/home/bruno/Scaricati/Dataset/GazeCapture/<SAMPLE>/"
json_file = dir + 'appleFace.json'  # Replace with the path to your JSON file
images_dir = dir + 'frames'       # Directory containing input images

utils.draw_bounding_boxes_opencv(json_file, images_dir, True)