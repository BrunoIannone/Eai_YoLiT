
# README

## Dataset Sample File Structure

To download the dataset, request the link here: [GazeCapture Download](https://gazecapture.csail.mit.edu/download.php)

Each sample will have the following structure:

```
Sample Folder/
    ├── frames/
    │   ├── [Images of the sample]
    ├── appleFace.json        # JSON containing face bounding box information for each sample image
    ├── appleLeftEye.json     # JSON containing left eye bounding box information for each sample image
    ├── appleRightEye.json    # JSON containing right eye bounding box information for each sample image
```
To convert this dataset for YoLo training, one might use the function **`convert_gazecapture_for_yolo`**. See [Dataset Management and Organization](#dataset-management-and-organization)
  
## Function in utils.py

This script (`utils.py`) contains a set of functions for managing and processing bounding boxes in images, primarily for datasets like GazeCapture. The functions allow extracting, converting, drawing, and organizing bounding boxes from JSON and TXT files, as well as providing tools for preparing datasets for training with YOLO.

## Main Functions

### Bounding Box Extraction and Management

- **`get_bbox_from_json(json_file)`**: Extracts bounding box coordinates from a JSON file and returns a list of dictionaries containing X, Y, W, H.
- **`get_bbox_from_txt(path)`**: Reads a TXT file containing bounding boxes and returns a list of coordinates.
- **`bbox_coord_from_dict(bbox_dict)`**: Converts a dictionary containing bounding box parameters into a tuple of integers.

### Drawing Bounding Boxes on Images

- **`draw_bounding_boxes_from_json(f_json_file, l_json_file, r_json_file, images_dir, show)`**: Draws face and eye bounding boxes on images read from JSON.
- **`draw_bounding_boxes_from_list(f_list, images_dir)`**: Draws face and eye bounding boxes on an image from a list of bounding boxes.

### Bounding Box Conversion and Formatting

- **`bbox_dict_to_list_of_list(bbox_dict)`**: Converts a list of bounding box dictionaries into a list of lists.
- **`coco_to_yolo_bbox_converter(bbox_list, image_width, image_height)`**: Converts bounding boxes from COCO format to YOLO format.

### Dataset Management and Organization

- **`sort_samples(images_dir, dest_dir)`**: Organizes images into training, validation, and test folders based on information in the `info.json` file.
- **`convert_gazecapture_for_yolo(src_folder)`**: Converts the GazeCapture dataset to be compatible with YOLO by organizing images and labels into structured folders.
- **`count_samples(src_folder)`**: Counts the number of valid images for each set (train, val, test).

### Writing Bounding Boxes to File

- **`write_list(elem_list, dest_folder, file_name)`**: Writes bounding boxes to a TXT file for YOLO training.

## ToDO
-to install distil yolo: pip3 install autodistill-grounded-sam-2 autodistill-yolov8 ( se dà un errore su cuda_home, dare sudo apt install nvidia-cuda-toolkit)
