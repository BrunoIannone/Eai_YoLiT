import cv2
import json
import os

def process_json(json_file):
    """Create a list of dictionaries containing bounding box coordinates for each valid sample. Invalid samples are represented with a 0. An example: [{'X': 112.28, 'Y': 188.28, 'W': 293.44, 'H': 293.44},0]

    Args:
        json_file (str): path to json file containing X and Y coordinates, Width (W) and Height (H) of the bounding box and a binary list of valid samples.

    Returns:
       List: List of dictionary for bounding box coordinates. Invalid samples are represented with a 0
    """
    

    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Ensure all keys are present in the JSON structure
    keys = ['X', 'Y', 'W', 'H', 'IsValid']
    if not all(key in data for key in keys):
        print("Error: Missing required keys in the JSON file.")
        return

    # Extract valid bounding boxes
    valid_boxes = []
    for i in range(len(data['IsValid'])):
        if data['IsValid'][i] == 1:  # Check if the detection is valid
            box = {
                'X': data['X'][i],
                'Y': data['Y'][i],
                'W': data['W'][i],
                'H': data['H'][i]
            }
            valid_boxes.append(box)
            
        else:
            valid_boxes.append(0)

    return valid_boxes

def draw_bounding_boxes_opencv(json_file, images_dir,show):
    """Draw the bounding boxes over all images of a sample. 

    Args:
        json_file (str): path to json file containing X and Y coordinates, Width (W) and Height (H) of the bounding box and a binary list of valid samples.
        images_dir (str): path to folder containing all sample images.
        show (bool): True: show all sample images one at a time (press a button to move to the next one). Else: do nothing.

    Raises:
        ValueError: raise an error when the number of images and bounding boxes mismatch.
    """
    data = process_json(json_file)
    images = sorted([img for img in os.listdir(images_dir)])
    
    if len(data) != len(images):
        
        raise ValueError(f"Dimension mismatch: data has {len(data)} items, but frames dir {images_dir} has {len(images)} items.")

    i = 0

    for img in images:
        print(f"index {i}, image {img}")

        if data[i] == 0:
            i+=1
            continue

        x = int(data[i]['X'])
        y = int(data[i]['Y'])
        w = int(data[i]['W'])
        h = int(data[i]['H'])
        
        sample = os.path.join(images_dir, f"{img}")

        sample = cv2.imread(sample)

        if sample is not None:

            # Draw the bounding box (top-left to bottom-right)
            start_point = (x, y)
            end_point = (x + w, y + h)
            color = (0, 0, 255)  # Red color in BGR
            thickness = 2

            # Draw the rectangle on the image
            cv2.rectangle(sample, start_point, end_point, color, thickness)
            if show:

                cv2.imshow("Sample",sample)
                k = cv2.waitKey(0)
                i+=1

            else:
                i+=1
        else:
            print(f"Failed to read {img}.")
            i+=1