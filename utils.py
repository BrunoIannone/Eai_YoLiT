import cv2
import json
import os
import shutil


def get_bbox_from_json(json_file):
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


def get_bbox_from_txt(path):
    """Takes a .txt file containing all the bounding boxes for an image and returns a list of them.
    .txt files is expected to have a row for each bounding box .

    Returns:
        List: returns a List of list, each containing a bounding box parameters. 
    """
    righe = []
    try:
        with open(path, 'r', encoding='utf-8') as file:
            righe = file.readlines()
    except FileNotFoundError:
        print(f"Errore: Il file '{path}' non è stato trovato.")
    except Exception as e:
        print(f"Errore durante la lettura del file: {e}")

    return [riga.strip() for riga in righe]


def bbox_coord_from_dict(bbox_dict):
    """Takes a dictionary of the form {X:,Y:,W:,H:} and returns the bbox_dict values as integers

    Args:
        bbox_dict (_type_): _description_

    Returns:
        tuple: tuple containing (x,y,w,h). Each element is an integer
    """
    x = int(bbox_dict['X'])
    y = int(bbox_dict['Y'])
    w = int(bbox_dict['W'])
    h = int(bbox_dict['H'])
    return x, y, w, h


def draw_bounding_boxes_from_json(f_json_file, l_json_file, r_json_file, images_dir, show):
    """Draw the bounding boxes over all images of a sample. Json files contain X and Y coordinates, Width (W) and Height (H) of the bounding box and a binary list of valid samples.

    Args:
        f_json_file (str): Path to face bounding box json
        l_json_file (str): Path to left eye bounding box json
        r_json_file (str): Path to right eye bounding box json

        images_dir (str): path to folder containing all sample images.
        show (bool): True: show all sample images one at a time (press a button to move to the next one). Else: do nothing.

    Raises:
        ValueError: raise an error when the number of images and bounding boxes mismatch.
    """
    f_data = get_bbox_from_json(f_json_file)

    l_eye_data = get_bbox_from_json(l_json_file)

    r_eye_data = get_bbox_from_json(r_json_file)

    images = sorted([img for img in os.listdir(images_dir)])

    if len(f_data) != len(images) or len(l_eye_data) != len(images) or len(r_eye_data) != len(images):
        raise ValueError(
            f"Dimension mismatch: data has {len(f_data)} items, left eye data has {len(l_eye_data)}, right eye data has {len(r_eye_data)}, frames dir {images_dir} has {len(images)} items.")

    for i, img in enumerate(images):

        print(f"index {i}, image {img}")

        if f_data[i] == 0 or l_eye_data[i] == 0 or r_eye_data[i] == 0:
            continue

        x = int(f_data[i]['X'])
        y = int(f_data[i]['Y'])
        w = int(f_data[i]['W'])
        h = int(f_data[i]['H'])

        l_x = int(l_eye_data[i]['X'])
        l_y = int(l_eye_data[i]['Y'])
        l_w = int(l_eye_data[i]['W'])
        l_h = int(l_eye_data[i]['H'])

        r_x = int(r_eye_data[i]['X'])
        r_y = int(r_eye_data[i]['Y'])
        r_w = int(r_eye_data[i]['W'])
        r_h = int(r_eye_data[i]['H'])

        sample = os.path.join(images_dir, f"{img}")
        sample = cv2.imread(sample)

        if sample is not None:

            # Face bbox
            start_point = (x, y)
            end_point = (x + w, y + h)
            color = (0, 0, 255)  # Red color in BGR
            thickness = 2
            cv2.rectangle(sample, start_point, end_point, color, thickness)

            # Left eye bbox
            l_start_point = (x + l_x, y + l_y)
            # eye bbox is referred to face bbox top left corner
            l_end_point = (x + l_w + l_x, y + l_h + l_y)
            color = (0, 255, 0)  # green square
            cv2.rectangle(sample, l_start_point, l_end_point, color, thickness)

            # right eye bbox
            r_start_point = (x + r_x, y + r_y)
            # eye bbox is referred to face bbox top left corner
            r_end_point = (x + r_w + r_x, y + r_h + r_y)
            color = (255, 0, 0)  # blue square
            cv2.rectangle(sample, r_start_point, r_end_point, color, thickness)

            if show:
                cv2.imshow("Sample", sample)
                k = cv2.waitKey(0)

        else:
            print(f"Failed to read {img}.")


# f_json_file, l_json_file, r_json_file, images_dir, show):
def draw_bounding_boxes_from_list(f_list, images_dir):
    """Draw the face, left eye and right eye bounding boxes on the image from a list"

    Args:
        f_list (List): List of list bboxes for face, left eye and right eye
        images_dir (str): path to the image

    """

    x = int(float(f_list[0].split(" ")[0]))
    y = int(float(f_list[0].split(" ")[1]))
    w = int(float(f_list[0].split(" ")[2]))
    h = int(float(f_list[0].split(" ")[3]))

    l_x = int(float(f_list[1].split(" ")[0]))
    l_y = int(float(f_list[1].split(" ")[1]))
    l_w = int(float(f_list[1].split(" ")[2]))
    l_h = int(float(f_list[1].split(" ")[3]))

    r_x = int(float(f_list[2].split(" ")[0]))
    r_y = int(float(f_list[2].split(" ")[1]))
    r_w = int(float(f_list[2].split(" ")[2]))
    r_h = int(float(f_list[2].split(" ")[3]))

    sample = cv2.imread(images_dir)

    if sample is not None:

        # Face bbox
        start_point = (x, y)
        end_point = (x + w, y + h)
        color = (0, 0, 255)  # Red color in BGR
        thickness = 2
        cv2.rectangle(sample, start_point, end_point, color, thickness)

        # Left eye bbox
        l_start_point = (x + l_x, y + l_y)
        # eye bbox is referred to face bbox top left corner
        l_end_point = (x + l_w + l_x, y + l_h + l_y)
        color = (0, 255, 0)  # green square
        cv2.rectangle(sample, l_start_point, l_end_point, color, thickness)

        # right eye bbox
        r_start_point = (x + r_x, y + r_y)
        # eye bbox is referred to face bbox top left corner
        r_end_point = (x + r_w + r_x, y + r_h + r_y)
        color = (255, 0, 0)  # blue square
        cv2.rectangle(sample, r_start_point, r_end_point, color, thickness)

        cv2.imshow("Sample", sample)
        k = cv2.waitKey(0)

    else:
        print(f"Failed to read {images_dir}.")


def sort_samples(images_dir, dest_dir):
    """Util function to sort images in train,val and test set. Note: dest_dir must be outside images_dir

    Args:
        images_dir (str): Images direction
        dest_dir (str): Destination path

    """
    samples = sorted([sample_folder for sample_folder in os.listdir(
        images_dir) if os.path.isdir(os.path.join(images_dir, sample_folder))])

    for i, img in enumerate(samples):
        print(img)

        with open(images_dir + img + "/info.json", 'r') as f:
            data = json.load(f)

        set = data["Dataset"]

        if set == "test":
            shutil.move(images_dir + img, dest_dir + "test")
        elif set == "val":
            shutil.move(images_dir + img, dest_dir + "val")
        else:
            shutil.move(images_dir + img, dest_dir + "train")


def bbox_dict_to_list_of_list(bbox_dict):
    """Converts a list of bounding box dictionaries into a list of lists.

    Args:
        bbox_dict (List): List of bounding boxes dictionaries. 

    Returns:
        List: List of list of bounding boxes.
    """
    bbox_list = []
    for dict in bbox_dict:
        if dict == 0:
            bbox_list.append(0)
        else:
            bbox_list.append(list(dict.values()))

    return bbox_list


def coco_to_yolo_bbox_converter(bbox_list, image_width, image_height):
    """
    Converte una lista di bounding box dal formato COCO al formato YOLO.

    Args:
        bboxes (list of tuples): Lista di bounding box in formato COCO (x_min, y_min, width, height).
        image_width (int): Larghezza dell'immagine.
        image_height (int): Altezza dell'immagine.

    Returns:
        list of tuples: Lista di bounding box in formato YOLO (x_center, y_center, width, height).
    """
    yolo_bboxes = []
    for bbox in bbox_list:
        x_min, y_min, width, height = bbox

        # Calcolo centro della bounding box
        x_center = x_min + width / 2.0
        y_center = y_min + height / 2.0

        # Normalizzazione rispetto alla dimensione dell'immagine
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        # Aggiungi la bounding box YOLO alla lista
        yolo_bboxes.append([x_center, y_center, width, height])

    return yolo_bboxes


def write_list(elem_list, dest_folder, file_name):
    """Aux function for writing all the bounding boxes (in elem_list) for an image in a .txt file. The .txt file will have a row for each class.

    Args:
        elem_list (List): List of list of bounding boxes parameters.
        dest_folder (str): path where we want to save the .txt.
        file_name (str): Name we want to give to the .txt
    """
    with open(os.path.join(dest_folder, file_name + ".txt"), "a") as file:

        elem_list = list(elem_list.values())
        line = " ".join(map(str, elem_list))
        file.write(line + "\n")


def convert_gazecapture_for_yolo(src_folder):
    """This function takes the GazeCapture dataset and make it suitable for YoLo training. 
    Namely, it creates the images and labels folders and, inside bot, it creates the train, val and test set folders.
    Next, each image is sorted inside its respective set folder while doing the same with a .txt file containing bbox information for each sample image. Notice that the .txt is put
    inside the labels folder. Each images and .txt have the same name. It is formed by the sample id plus an index representing the i-th image for the sample.
    Namely, each file will be <sampleid>_<index>.{jpg,txt}


    Args:
        src_folder (str): Path to dataset folder

    Raises:
        ValueError: raise an error if bounding boxes number for an image mismatch
    """

    os.makedirs(os.path.join(src_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(src_folder, "images/test"), exist_ok=True)
    os.makedirs(os.path.join(src_folder, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(src_folder, "images/train"), exist_ok=True)

    os.makedirs(os.path.join(src_folder, "labels"), exist_ok=True)
    os.makedirs(os.path.join(src_folder, "labels/test"), exist_ok=True)
    os.makedirs(os.path.join(src_folder, "labels/val"), exist_ok=True)
    os.makedirs(os.path.join(src_folder, "labels/train"), exist_ok=True)

    to_avoid_dirs = ["images", "labels"]

    for persona in os.listdir(src_folder):

        if persona in to_avoid_dirs or not os.path.isdir(os.path.join(src_folder, persona)):
            continue

        f_data = get_bbox_from_json(os.path.join(
            src_folder, persona, 'appleFace.json'))

        l_eye_data = get_bbox_from_json(os.path.join(
            src_folder, persona, 'appleLeftEye.json'))

        r_eye_data = get_bbox_from_json(os.path.join(
            src_folder, persona, 'appleRightEye.json'))

        with open(os.path.join(src_folder, persona, "info.json"), 'r') as f:
            destination_set = json.load(f)

        if len(f_data) != len(r_eye_data) or len(f_data) != len(l_eye_data):
            raise ValueError(
                f"Dimension mismatch: data has {len(f_data)} items, left eye data has {len(l_eye_data)}, right eye data has {len(r_eye_data)}")

        samples = sorted([sample_folder for sample_folder in os.listdir(
            os.path.join(src_folder, persona, "frames"))])

        for i, image in enumerate(samples):
            if f_data[i] == 0 or l_eye_data[i] == 0 or r_eye_data[i] == 0:
                # print("Invalid bbox", persona, image)
                continue

            src_image_path = os.path.join(src_folder, persona, "frames", image)
            new_name = persona + "_" + str(i) + "." + "jpg"

            img_dest_folder = os.path.join(
                src_folder, "images", destination_set["Dataset"])
            shutil.move(src_image_path, img_dest_folder)
            os.rename(os.path.join(img_dest_folder, image),
                      os.path.join(img_dest_folder, new_name))
            txt_dest_folder = os.path.join(
                src_folder, "labels", destination_set["Dataset"])
            write_list(f_data[i], txt_dest_folder, new_name.split(".")[0])
            write_list(l_eye_data[i], txt_dest_folder, new_name.split(".")[0])
            write_list(r_eye_data[i], txt_dest_folder, new_name.split(".")[0])
        shutil.rmtree(os.path.join(src_folder, persona))


def count_samples(src_folder):
    """Count the number of valid images (so eye bbox, left eye bbox and right eye bbox "isValid" parameter in the .json must be 1) for each person.
    The counting is divided for train, validation and test set. This info is get from sample/info.json.

    Args:
        src_folder (str): dataset folder path
    """
    set = {"test": 0, "train": 0, "val": 0}
    for persona in os.listdir(src_folder):
        if not os.path.isdir(os.path.join(src_folder, persona)):
            continue
        f_data = get_bbox_from_json(os.path.join(
            src_folder, persona, 'appleFace.json'))

        l_eye_data = get_bbox_from_json(os.path.join(
            src_folder, persona, 'appleLeftEye.json'))

        r_eye_data = get_bbox_from_json(os.path.join(
            src_folder, persona, 'appleRightEye.json'))

        samples = sorted([sample_folder for sample_folder in os.listdir(
            os.path.join(src_folder, persona, "frames"))])
        with open(os.path.join(src_folder, persona, "info.json"), 'r') as f:
            destination_set = json.load(f)

        for i, image in enumerate(samples):
            if f_data[i] == 0 or l_eye_data[i] == 0 or r_eye_data[i] == 0:
                # print("Invalid bbox", persona, image)
                continue
            set[destination_set["Dataset"]] += 1

    print(set)
