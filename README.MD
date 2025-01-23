
to download the dataset:
wget -O gazecapture.tar "https://gazecapture.csail.mit.edu/dataset.php?email=iannone.1883472@studenti.uniroma1.it&hash=e744f91c29ec99f0e662c9177946c627"
to install distil yolo: pip3 install autodistill-grounded-sam-2 autodistill-yolov8 ( se dà un errore su cuda_home, dare sudo apt install nvidia-cuda-toolkit)


per recuperare tutte le bounding box valide di un sample: 

   
dir = "/home/bruno/Scaricati/Dataset/GazeCapture/<SAMPLE>/"
json_file = dir + 'appleFace.json'  # Replace with the path to your JSON file
images_dir = dir + 'frames'       # Directory containing input images

utils.draw_bounding_boxes_opencv(json_file, images_dir, True)

To sort the dataset in train/val/test sets, once extracted, you can use the sort_samples method. Notice, destination folders must be outside the GazeCapture Folder; you can move it back inside after its execution.

Folder structure for a sample:
-Folder
    -train/val/test
        -Sample folder (id = number)
            -frames
                -Images for the sample (sorted in ascending order) (.jpeg) (id = number)
            -appleFace.json (Json containing faces bounding boxes information for each sample image)