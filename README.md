# IRP-Eye-out

This project done for my individual research project allow to generate data from thanks to Airsim, train and evaluate the model
The best configuration model is available in the config_file folder

## Installation

Because the project is using Airsim, there are many dependencies to install

Install Unreal engine through the [EpicGame Store](https://store.epicgames.com/en-US/) (This project worked with Unreal v4.27)
Install [airsim](https://microsoft.github.io/AirSim/build_windows/) and build the package
Create a new C++ project in Unreal and add the built Airsim plugin into it. You can follow this [tutorial](https://www.microsoft.com/en-us/research/video/unreal-airsim-setup-from-scratch/)
Change the setting.json from Airsim to the one in the config_file folder
For the Map there are some free assets available as Landscape Mountain, DesertRally Race or AccuCity in the EpicGame Store for Unreal, don't forget to change the game mode to "AirsimGamemode" before pressing play on the Engine.


For Python, install the requirement.txt and be careful on the Pytorch version to be compatible with your own cuda version
```bash
pip install -r requirements.txt
```

## Usage

### Data generation

Here is an example on how to generate data

```bash
from pathlib import Path
from src import data_generation

vehicle_dict = {1: {"Name": "Balloon1",
                    "PawnPath": "Balloon1",
                    "Mesh": "test_ballon1",
                    "Max_dist_m": 4000,
                    "Min_dist_m": 300,
                    "yolo_class":'2'},
                2: {"Name": "Ju-87",
                    "PawnPath": "Ju-87",
                    "Mesh": "ju87_body",
                    "Max_dist_m": 3000,
                    "Min_dist_m": 300,
                    "yolo_class":'0'},
                3: {"Name": "Piper",
                    "PawnPath": "Piper",
                    "Mesh": "piper_pa18",
                    "Max_dist_m": 3000,
                    "Min_dist_m": 300,
                    "yolo_class":'0'},
                4:{"Name": "AW101",
                    "PawnPath": "AW101",
                    "Mesh": "AW101_AW101",
                    "Max_dist_m": 3000,
                    "Min_dist_m": 300,
                    "yolo_class":'1'},
                5:{"Name": "Agusta",
                    "PawnPath": "Agusta",
                    "Mesh": "Agusta_fused",
                    "Max_dist_m": 3000,
                    "Min_dist_m": 300,
                    "yolo_class":'1'},
                6:{"Name": "Balloon2",
                    "PawnPath": "Balloon2",
                    "Mesh": "balloon2",
                    "Max_dist_m": 4000,
                    "Min_dist_m": 300,
                    "yolo_class":'2'}}

used_list = ["Balloon1","Balloon2","Ju-87","Piper","Agusta","AW101"]
dist_interval =[1000,4000]
camera_angle = ["front", "back", "right", "left"]
level_list = ["LandscapeMap"]
weather_dict = {"3AM":"2023-06-15 11:15:00",
                "6AM":"2023-06-15 14:15:00",
                "9AM":"2023-06-15 17:15:00",
                "12AM":"2023-06-15 20:15:00",
                "3PM":"2023-06-15 23:15:00",
                "9PM":"2023-06-15 5:15:00",
                "12PM":"2023-06-15 8:15:00",
                "6PM":"2023-06-15 2:15:00"}
path = Path("data/image/data_1000min_4000max")

#generate data
data_generation.dataset_generation(path, vehicle_dict, used_list, 600, camera_angle, level_list,
                                   weather_dict, dist_interval)
#order normal data
data_generation.order_dataset(path, dist_interval)

#order and create split sliced, normal data must be ordered before running this
data_generation.order_slice_data(path / ("ordered_data_" + str(dist_interval) + ".json"), 4, 0.2, dist_interval)

#generate split
data_generation.generate_train_val_test(path / ("ordered_data_" + str(dist_interval) + ".json"), (80, 10, 10), "", 42)
data_generation.generate_train_val_test(path / ("ordered_data_slice_" + str(dist_interval) + ".json"), (80, 10, 10), "_sliced", 42)
```

### Training

Here is how to train a Yolov8 model :

Change the config file with the correct parrameters
```bash
from src.detection import training

training("config_files/yolov8_config.yaml")
```

### Evaluate

Here is how to evaluate your model :

```bash
from pathlib import Path
import math
from src.detection import evaluate_model, generate_graph

resolution = [4608, 2592]
evaluation_df = evaluate_model(Path("yolov8s.pt"),Path("data/image/test_readme/test/images"),Path("data\image\\test_readme\sim_log.json"),
               "SAHI", 0.3, [math.ceil(resolution[0]/2* 1.2),math.ceil(resolution[1]/2 * 1.2)],
               [0.2,0.2],0.3,'cuda:0',True)
generate_graph(evaluation_df,
                   [{"x": "Pred_result", "y": "Distance"},
                    {"x": "Pred_result", "y": "Timeoftheday"},
                    {"x": "Pred_result", "y": "Level"},
                    {"x": "Pred_result","y":"Weather_param"},
                    {"x": "Duplicate?","y":"Pred_result"}])
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
