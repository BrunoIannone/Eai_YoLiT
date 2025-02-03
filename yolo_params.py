dir = "/home/bruno/Scaricati/Dataset/GazeCapture"  # 03454

yolo_params_ = {
        "data": dir + "/data.yaml", 
        # "fraction":1.0, #Allows for training on a subset of the full dataset
        "epochs": 5, 
        "imgsz": 640,
        "batch":-1,
        "lr0":1e-3,
        "patience":2,
        "save_period":-1,
        "workers": 2,
        "optimizer":"AdamW",
        "freeze":10,
        "dropout":0.2,
        "warmup_epochs":2,
        "plots": True,
        ##AUGMENTATION##
        "degrees":90.0,
        "perspective":0.0001,
        "fliplr":0.0,
        "mosaic":0.0,

        }