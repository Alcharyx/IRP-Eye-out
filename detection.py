import airsim
import ultralytics
import pandas
from ultralytics import YOLO
from ultralytics.engine.results import Results
from sahi.prediction import PredictionResult
import os
import json
import math
import numpy as np
import polars as pl
from pathlib import Path
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi import AutoDetectionModel
import matplotlib.pyplot as plt
from tqdm import tqdm


def training(config):
    """
    Train yolo model based on .yaml configuration file
    Parameters
    ----------
    config, str :
        String path to the .yaml config file

    Returns
    -------

    """
    model = YOLO()
    model.train(cfg=config)


def predict_sahi(weight_path, list_img, slice, overlap, conf,
                 device):  # batch_size manually changed to 4 for faster result
    """
    Predict using sahi based on given parameters
    Parameters
    ----------
    weight_path, str :
        String of the path for the model weights
    list_img, list[Path] :
        List of the path of each images
    slice, list[int, int] :
        List containing width and height of a slice (without overlap)
    overlap, list[int,int] :
        List containing width and height ratio of the overlap
    conf, float
        confidence threshold to count the prediction as valid
    device, int or str :
        define the device use either CPU or GPU

    Returns
    -------

    """
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=weight_path,
        confidence_threshold=conf,
        device=device
    )
    result = {}
    for i in tqdm(list_img):
        result[i.stem] = (get_sliced_prediction(str(i),
                                                sahi_model,
                                                slice_width=slice[0],
                                                slice_height=slice[1],
                                                overlap_width_ratio=overlap[0],
                                                overlap_height_ratio=overlap[1],
                                                verbose=0),)
    return result


def get_img_log(img_name, log_path):
    """
    Load the data of one image in the log file
    Parameters
    ----------
    img_name, str :
        name of the image
    log_path, Path :
        Path of the log file

    Returns, dict :
        Dict containing the image parameters and data in the simulation
    -------

    """
    sim_number, frame_number, _ = img_name.split("_")
    sim_number = str(int(sim_number))
    frame_number = str(int(frame_number))
    with open(log_path) as json_file:
        sim_log = json.load(json_file)
    result_dict = sim_log[sim_number]["frame_data"][frame_number]
    dict_items = list(result_dict.items())
    dict_items.insert(2, ("weather", sim_log[sim_number]["weather"]))

    return dict(dict_items)


def reformat_predict(result_dict):
    """
    Change result to common result dict format between Yolov8 and SAHI
    Parameters
    ----------
    result_dict, result from Prediction (SAHI or normal YOLO)

    Returns, dict :
        Common result format
    -------

    """
    # bbx format xmax,ymax,xmin,ymin
    return_dict = {}
    counter = 1
    if type(result_dict) == Results:
        for pred in result_dict.boxes.data.tolist():
            return_dict[counter] = {"class": int(pred[5]),
                                    "bbx": [pred[2], pred[3], pred[0], pred[1]],
                                    "conf": pred[4]}
            counter += 1
    elif type(result_dict[0]) == PredictionResult:
        for pred in result_dict[0].object_prediction_list:
            return_dict[counter] = {"class": pred.category.id,
                                    "bbx": [pred.bbox.maxx, pred.bbox.maxy,
                                            pred.bbox.minx, pred.bbox.miny],
                                    "conf": pred.score.value}
            counter += 1
    return return_dict


def compute_iou(box1, box2):
    """
    Compute the IoU between 2 bbx
    Parameters
    ----------
    box1, list[int] :
        [xmax, ymax, xmin, ymin]
    box2, list[int] :
        [xmax, ymax, xmin, ymin]

    Returns
    -------

    """
    xmax1, ymax1, xmin1, ymin1 = box1
    xmax2, ymax2, xmin2, ymin2 = box2
    x_left = max(xmin1, xmin2)
    y_top = max(ymin1, ymin2)
    x_right = min(xmax1, xmax2)
    y_bottom = min(ymax1, ymax2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def add_to_df2(df, img_log, pred_result, img_path, IOU_threshold):
    """
    Add a row to the polars dataframe depending on the prediction result
    Parameters
    ----------
    df, Polars dataframe :
        The current polars dataframe
    img_log, dict :
        Dict containing the image data from the log
    pred_result, dict :
        Prediction data made by the model
    img_path, Path :
        Image path
    IOU_threshold, float :
        Value of the IoU threshold to be considered a valid prediction

    Returns
    -------

    """
    angle = img_path.stem.split("_")[2]
    list_key = [key for key in list(img_log.keys())[4:] if key.split("_")[0] == angle]
    pred_per_key = {}
    for key in list_key:
        pred_per_key[key] = 0
    for pred_key in pred_result.keys():
        max_IOU = -1
        best_key = None
        for key in list_key:
            truth_box = [img_log[key]["label"]["box2D"]["x_max"],
                         img_log[key]["label"]["box2D"]["y_max"],
                         img_log[key]["label"]["box2D"]["x_min"],
                         img_log[key]["label"]["box2D"]["y_min"]]
            key_IOU = compute_iou(pred_result[pred_key]["bbx"], truth_box)
            if max_IOU < key_IOU:
                max_IOU = key_IOU
                best_key = key
        if max_IOU >= IOU_threshold:
            # save as TP
            truth_box = [img_log[best_key]["label"]["box2D"]["x_max"],
                         img_log[best_key]["label"]["box2D"]["y_max"],
                         img_log[best_key]["label"]["box2D"]["x_min"],
                         img_log[best_key]["label"]["box2D"]["y_min"]]

            new_img_row_data = {
                'Img_path': [str(img_path)],
                'Weather_param': [img_log["weather"]["param"]],
                'Weather_value': [img_log["weather"]["val"]],
                'Level': [img_log["currentlevel"]],
                'Timeoftheday': [img_log["timeoftheday"]],
                'Yolo_class_truth': [img_log[best_key]["label"]["yolo_class"]],
                'Yolo_class_pred': [pred_result[pred_key]["class"]],
                'Conf': [pred_result[pred_key]["conf"]],
                'Object_name': [img_log[best_key]["label"]["name"]],
                'Box2D_pred': [pred_result[pred_key]["bbx"]],
                'Box2D_truth': [truth_box],
                'Pred_result': ["TP"],
                'Duplicate?': [False],
                'Distance': [img_log[best_key]["label"]["distance"]]
            }
            pred_per_key[best_key] += 1
        else:
            # save as FP
            new_img_row_data = {
                'Img_path': [str(img_path)],
                'Weather_param': [img_log["weather"]["param"]],
                'Weather_value': [img_log["weather"]["val"]],
                'Level': [img_log["currentlevel"]],
                'Timeoftheday': [img_log["timeoftheday"]],
                'Yolo_class_truth': [None],
                'Yolo_class_pred': [pred_result[pred_key]["class"]],
                'Conf': [pred_result[pred_key]["conf"]],
                'Object_name': [None],
                'Box2D_pred': [pred_result[pred_key]["bbx"]],
                'Box2D_truth': [None],
                'Pred_result': ["FP"],
                'Duplicate?': [False],
                'Distance': [None]
            }
        new_img_row = pl.DataFrame(new_img_row_data, schema={'Img_path': pl.datatypes.Utf8,
                                                             'Weather_param': pl.datatypes.Int32,
                                                             'Weather_value': pl.datatypes.Float32,
                                                             'Level': pl.datatypes.Utf8,
                                                             'Timeoftheday': pl.datatypes.Utf8,
                                                             'Yolo_class_truth': pl.datatypes.Int32,
                                                             'Yolo_class_pred': pl.datatypes.Int32,
                                                             'Conf': pl.datatypes.Float32,
                                                             'Object_name': pl.datatypes.Utf8,
                                                             'Box2D_pred': pl.datatypes.List(
                                                                 pl.datatypes.Float32),
                                                             'Box2D_truth': pl.datatypes.List(
                                                                 pl.datatypes.Float32),
                                                             'Pred_result': pl.datatypes.Utf8,
                                                             'Duplicate?': pl.datatypes.Boolean,
                                                             'Distance': pl.datatypes.Float32})
        df = df.extend(new_img_row)
    for count_key in pred_per_key.keys():
        if pred_per_key[count_key] == 0:
            # save as FN
            truth_box = [img_log[count_key]["label"]["box2D"]["x_max"],
                         img_log[count_key]["label"]["box2D"]["y_max"],
                         img_log[count_key]["label"]["box2D"]["x_min"],
                         img_log[count_key]["label"]["box2D"]["y_min"]]
            new_img_row_data = {
                'Img_path': [str(img_path)],
                'Weather_param': [img_log["weather"]["param"]],
                'Weather_value': [img_log["weather"]["val"]],
                'Level': [img_log["currentlevel"]],
                'Timeoftheday': [img_log["timeoftheday"]],
                'Yolo_class_truth': [img_log[count_key]["label"]["yolo_class"]],
                'Yolo_class_pred': [None],
                'Conf': [None],
                'Object_name': [img_log[count_key]["label"]["name"]],
                'Box2D_pred': [None],
                'Box2D_truth': [truth_box],
                'Pred_result': ["FN"],
                'Duplicate?': [False],
                'Distance': [img_log[count_key]["label"]["distance"]]
            }
            new_img_row = pl.DataFrame(new_img_row_data, schema={'Img_path': pl.datatypes.Utf8,
                                                                 'Weather_param': pl.datatypes.Int32,
                                                                 'Weather_value': pl.datatypes.Float32,
                                                                 'Level': pl.datatypes.Utf8,
                                                                 'Timeoftheday': pl.datatypes.Utf8,
                                                                 'Yolo_class_truth': pl.datatypes.Int32,
                                                                 'Yolo_class_pred': pl.datatypes.Int32,
                                                                 'Conf': pl.datatypes.Float32,
                                                                 'Object_name': pl.datatypes.Utf8,
                                                                 'Box2D_pred': pl.datatypes.List(
                                                                     pl.datatypes.Float32),
                                                                 'Box2D_truth': pl.datatypes.List(
                                                                     pl.datatypes.Float32),
                                                                 'Pred_result': pl.datatypes.Utf8,
                                                                 'Duplicate?': pl.datatypes.Boolean,
                                                                 'Distance': pl.datatypes.Float32})
            df = df.extend(new_img_row)
        elif pred_per_key[count_key] > 1:
            # duplicate pred ?
            conditions = (df["Img_path"] == str(img_path)) & (df["Pred_result"] == "TP") & (
                        df["Object_name"] == img_log[count_key]["label"]["name"])
            df = df.with_columns(pl.when(conditions).then(True).otherwise(df['Duplicate?']).alias('Duplicate?'))
    return df


def create_evaluation_df(data_path, reformed_results, log_path, conf, save_flag):
    """
    Creates the evaluation dataframe
    Parameters
    ----------
    data_path, Path :
        Path of the data
    reformed_results, Dict :
        Reformed_result dict
    log_path, Path :
        Path of the log data file
    conf, float :
        confidence threshold
    save_flag, bool :
        flag to save the df into a json

    Returns
    -------

    """

    data_dict = {
        'Img_path': [],
        'Weather_param': [],
        'Weather_value': [],
        'Level': [],
        'Timeoftheday': [],
        'Yolo_class_truth': [],
        'Yolo_class_pred': [],
        'Conf': [],
        'Object_name': [],
        'Box2D_pred': [],
        'Box2D_truth': [],
        'Pred_result': [],
        'Duplicate?': [],
        'Distance': []
    }
    df = pl.DataFrame(data_dict, schema={'Img_path': pl.datatypes.Utf8,
                                         'Weather_param': pl.datatypes.Int32,
                                         'Weather_value': pl.datatypes.Float32,
                                         'Level': pl.datatypes.Utf8,
                                         'Timeoftheday': pl.datatypes.Utf8,
                                         'Yolo_class_truth': pl.datatypes.Int32,
                                         'Yolo_class_pred': pl.datatypes.Int32,
                                         'Conf': pl.datatypes.Float32,
                                         'Object_name': pl.datatypes.Utf8,
                                         'Box2D_pred': pl.datatypes.List(pl.datatypes.Float32),
                                         'Box2D_truth': pl.datatypes.List(pl.datatypes.Float32),
                                         'Pred_result': pl.datatypes.Utf8,
                                         'Duplicate?': pl.datatypes.Boolean,
                                         'Distance': pl.datatypes.Float32})
    for img_path in tqdm(data_path.iterdir()):
        img_log = get_img_log(img_path.stem, log_path)
        df = add_to_df2(df, img_log, reformed_results[img_path.stem], img_path, conf)
    if save_flag:
        df.write_json(file=str(data_path.parent / 'test_data.json'), row_oriented=True, pretty=True)
    return df


def good_split_slice(number):
    """
    Split a number into the highest decomposition where a x b = number
    Parameters
    ----------
    number, int :
        Number you want to split in 2

    Returns a,b, int :
        highest decompostion number
    -------

    """
    if number == 1:
        return 1, 1
    a = int(number ** 0.5)
    while a > 0:
        if number % a == 0:
            b = number // a
            if a == 1:
                raise ValueError("The number of slice cannot be divided properly change the number of slice")
            else:
                return a, b
        a -= 1


def generate_graph(json_path, graphdict):
    """
    Generate graph with parameters depending on graphdict
    Parameters
    ----------
    json_path, Path :
        Path of the dataframe json
    graphdict, dic :
        dict containing x and y field for the graph

    Returns
    -------

    """
    df = pl.read_json(json_path)
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'brown']
    bar_width = 0.2
    for dict_axis in graphdict:

        unique_val_x = df[dict_axis["x"]].unique().to_list()
        unique_val_y = df[dict_axis["y"]].unique().to_list()

        dict_val = {}
        if dict_axis["y"] == "Distance":
            unique_val_y = [item for item in unique_val_y if item is not None]
            quotient = (round((max(unique_val_y) / 500)))
            max_range = quotient * 500
            list_axis = np.linspace(0, max_range, quotient + 1)
            unique_val_y = []
            for i in range(1, len(list_axis)):
                unique_val_y.append(str(int(list_axis[i - 1])) + "_" + str(int(list_axis[i])))
            for x_val in unique_val_x:
                dict_val[x_val] = []
                for y_val in unique_val_y:
                    lower_range, higher_range = y_val.split("_")
                    dict_val[x_val].append(df.filter(
                        (pl.col(dict_axis["y"]) > int(lower_range)) & (pl.col(dict_axis["y"]) <= int(higher_range)) & (
                                    pl.col(dict_axis["x"]) == x_val)).shape[0])
        else:
            for x_val in unique_val_x:
                dict_val[x_val] = []
                for y_val in unique_val_y:
                    dict_val[x_val].append(
                        df.filter((pl.col(dict_axis["y"]) == y_val) & (pl.col(dict_axis["x"]) == x_val)).shape[0])
        index = np.arange(len(unique_val_y))

        counter = 0
        print(sorted(dict_val.keys()))
        for i in sorted(dict_val.keys()):

            plt.bar(index + counter * bar_width, dict_val[i], bar_width, label=i, color=colors[counter])
            counter += 1
            if counter == 10:
                raise ValueError("More than 10 different parameter isn't supported for more readable plot")
        plt.xlabel('Parameters')
        plt.ylabel('Count')
        plt.title(dict_axis["y"] + " over " + dict_axis["x"])
        plt.xticks(index + bar_width, unique_val_y)

        plt.legend()
        plt.show()
