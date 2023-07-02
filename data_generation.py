import math
import time
from pathlib import Path
import shutil
from skimage import color
import random
import numpy as np
import airsim
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import cv2
import winsound


class travel:
    """
    Class with all the needed value to imitate a travel from point A to B
    """

    def __init__(self, start, end, speed, angle_start, angle_end):
        self.start_coordinates = start
        self.end_coordinates = end
        self.speed = speed
        self.angle_start = angle_start
        self.angle_end = angle_end


class WeatherParameter:
    """
    Class with corresponding value for Airsim API
    """
    Nothing = None
    Rain = 0
    # Roadwetness = 1
    Snow = 2
    # RoadSnow = 3
    # MapleLeaf = 4
    # RoadLeaf = 5
    Dust = 6
    Fog = 7  # very low visibility


def save_img(client, frame, camera_angle, save_path, sim_count, camera_vehicle, vehicle_dict):
    """
    Save images if a flying object have been detected and there is no occlusion
    Parameters
    ----------
    client, MultirotorClient :
        Airsim multirotor client
    frame, int :
        Number of the current frame in the simulation
    camera_angle, list :
        List of each camera angle used
    save_path, Path :
        Path were all the data folder will be stored
    sim_count, int :
        Number of the current simulation
    camera_vehicle, str :
        Name of the camera_vehicle in Airsim
    vehicle_dict, dict :
        Dictionary containing all the information needed per vehicle
                    
    Returns Label, dict:
        Dictionary containing all the bounding boxes data from the flying vehicle detected on the images for each camera
        angle
    -------
    """
    img_list = []
    for i in camera_angle:
        response = client.simGetImages([airsim.ImageRequest(i, airsim.ImageType.Scene, False, False)], "", False)
        # print("img ", i)
        img_list.append(response[0])
    label = checkbb_camera(client, camera_vehicle, camera_angle, vehicle_dict)
    if label:
        for i in range(len(camera_angle)):
            # print("save img ", camera_angle[i])
            img1d = np.frombuffer(img_list[i].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(img_list[i].height, img_list[i].width, 3)
            gray_image = color.rgb2gray(img_rgb) * 255
            frame_path = str(save_path / (
                        "{:02d}".format(sim_count) + '_' + "{:03d}".format(frame) + '_' + camera_angle[i] + '.png '))
            cv2.imwrite(frame_path, gray_image)
    return label


def create_vehicle(client, pawn_path, name):
    """
    Spawn a vehicle in airsim with a special name and pawn_path
    Parameters
    ----------
    client, MultirotorClient :
        Airsim multirotor client
    pawn_path, str :
        PawnPath of a vehicle defined in the Airsim settings.json
    name, str :
        Name of the vehicle for the simulation
        
    Returns Nothing
    -------

    """
    pose = airsim.Pose(airsim.Vector3r(0, 0, -2))
    client.simAddVehicle(name, "simpleflight", pose, pawn_path)
    client.enableApiControl(True, vehicle_name=name)


def teleport(client, vehicle_name, coordinates, orientation=(0, 0, 0)):
    """
    Teleport a vehicle to coordinates in free fall
    Parameters
    ----------
    client, MultirotorClient :
        Airsim multirotor client
    vehicle_name, str :
        Name of the vehicle in the simulation
    coordinates, list :
        List of coordinates in NED coordinates system (x,y,-z)
        
    Returns Nothing
    -------

    """
    pose = airsim.Pose(airsim.Vector3r(coordinates[0],
                                       coordinates[1],
                                       coordinates[2]),
                       airsim.to_quaternion(orientation[0],
                                            orientation[1],
                                            orientation[2]))
    client.simSetVehiclePose(pose, True, vehicle_name)
    # client.moveToZAsync(coordinates[2], 10,vehicle_name=vehicle_name).join()
    # client.takeoffAsync(vehicle_name=vehicle_name).join()
    # client.hoverAsync(vehicle_name).join()


def get_coordinates(center, radius, angle):
    """
    Find the 3D coordinates around a circle depending on the angle
    Parameters
    ----------
    center, list :
        List of the center coordinates
    radius, list :
        Value in meter for the size of the circle radius
    angle, float :
        Value in degree of the angle
        
    Returns coordinates, list :
        Coordinates on the circle at the specific angle
    -------

    """
    x = center[0] + radius * math.cos(math.radians(angle))
    y = center[1] + radius * math.sin(math.radians(angle))
    coordinates = [x, y, center[2]]
    return coordinates


def get_distance(coord1, coord2):
    """
    Get distance between 2  3D points
    Parameters
    ----------
    coord1, list :
        Coordinate of point 1
    coord2, list :
        Coordinate of point 2

    Returns distance, float :
        Distance between the 2 points
    -------

    """
    return math.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)


def generate_travel(center, radius, speed):
    """
    Randomly generate a starting point and ending point on a circle and saves it
    Parameters
    ----------
    center, list :
        Coordinate of the center of the circle
    radius, float :
        Radius of the circle
    speed, float :
        Speed of the object travelling

    Returns trip, Travel :
        Instance of the travel class
    -------

    """
    flag_farenough = False
    angle_start = random.randint(1, 360)
    start = get_coordinates(center, radius, angle_start)
    while not flag_farenough:
        angle_end = random.randint(0, 360)
        end = get_coordinates(center, radius, angle_end)
        distance = get_distance(start, end)
        # print(distance)
        if distance >= radius * 1.5:
            flag_farenough = True

    altitude_change = random.uniform(-1, 1) * center[2] * random.randint(5, 20) / 100
    start[2] = start[2] + round(altitude_change / 2)
    end[2] = end[2] - round(altitude_change / 2)
    trip = travel(start, end, speed, angle_start, angle_end)

    return trip


def estimate_trajectory(center, radius, cam_travel: travel, list_travel: list[travel], plot: bool, save_path: Path,
                        sim_count):
    """
    Estimate the trajectory of all the flying object of the simulation, plot it and saves it
    Parameters
    ----------
    center, list :
        Coordinate of the center of the circle
    radius, float :
        Radius of the circle
    cam_travel, Travel :
        Travel of the camera vehicle
    list_travel, list[Travel] :
        List of the travel of all the other vehicles
    plot, bool :
        Flag to allow plotting or not
    save_path, Path :
        Path were all the data folder will be stored
    sim_count, int :
        Number of the current simulation

    Returns Nothing
    -------

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Travel1
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = min(cam_travel.end_coordinates[2], cam_travel.start_coordinates[2]) * np.ones_like(theta)
    ax.plot(x, y, z, color='green')

    ax.quiver(
        cam_travel.start_coordinates[0], cam_travel.start_coordinates[1], cam_travel.start_coordinates[2],
        cam_travel.end_coordinates[0] - cam_travel.start_coordinates[0],
        cam_travel.end_coordinates[1] - cam_travel.start_coordinates[1],
        cam_travel.end_coordinates[2] - cam_travel.start_coordinates[2],
        arrow_length_ratio=0.1, color='red')

    # Travel2
    for travel2 in list_travel:
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.quiver(
            travel2.start_coordinates[0], travel2.start_coordinates[1], travel2.start_coordinates[2],
            travel2.end_coordinates[0] - travel2.start_coordinates[0],
            travel2.end_coordinates[1] - travel2.start_coordinates[1],
            travel2.end_coordinates[2] - travel2.start_coordinates[2],
            arrow_length_ratio=0.1, color='blue')
    plt.savefig(save_path / ("{:02d}".format(sim_count) + "_estimated_path.png"))
    if plot:
        plt.show()
    plt.close()


def log_dict(cam_travel, list_travel, weather_param, weather_val, vehicle_dict, frame_data):
    """
    Generate the log dictionary for the current simulation
    Parameters
    ----------
    cam_travel, Travel :
        Travel of the camera vehicle
    list_travel, list[Travel] :
        List of the travel of all the other vehicles
    weather_param, WeatherParameter :
        Value of the weather parameter for this simulation
    weather_val, float :
        Value for the intensity of the weather for this simulation
    vehicle_dict, dict :
        Dictionary containing all the information needed per vehicle
    frame_data, dict :
        Detected vehicle data for each frames

    Returns log, dict :
        Concatenated dict of all the data in the input
    -------

    """
    log = {}
    log["travel_cam"] = {"start": cam_travel.start_coordinates,
                         "end": cam_travel.end_coordinates,
                         "speed": cam_travel.speed,
                         "angle_start": cam_travel.angle_start,
                         "angle_end": cam_travel.angle_end
                         }
    for travel_id in range(0, len(list_travel)):
        list_key = list(vehicle_dict.keys())
        log["travel" + vehicle_dict[list_key[travel_id]]["Name"]] = {"start": list_travel[travel_id].start_coordinates,
                                                                     "end": list_travel[travel_id].end_coordinates,
                                                                     "speed": list_travel[travel_id].speed,
                                                                     "angle_start": list_travel[travel_id].angle_start,
                                                                     "angle_end": list_travel[travel_id].angle_end
                                                                     }
    log["weather"] = {"param": weather_param,
                      "val": weather_val}
    log["frame_data"] = frame_data
    return log


def change_weather(client: airsim.MultirotorClient, weather_param: WeatherParameter, weather_val):
    """
    Change the weather of the simulation by calling Airsim API
    Parameters
    ----------
    client, MultirotorClient :
        Airsim multirotor client
    weather_param, WeatherParameter :
        Value of the weather parameter for this simulation
    weather_val, float :
        Value for the intensity of the weather for this simulation

    Returns Nothing
    -------

    """
    if weather_param is not None:
        client.simEnableWeather(True)
        client.simSetWeatherParameter(weather_param, weather_val)


def set_next_position(client, travel: travel, vehicle_name, time_delay, orientation):
    """
    Set the next position of a vehicle on the way of its travel
    Parameters
    ----------
    client, MultirotorClient :
        Airsim multirotor client
    travel, Travel :
        Travel of the vehicle
    vehicle_name, str :
        Name of the vehicle in the simulation
    time_delay, float :
        Time expected of the delay between the current position and the next position
    orientation, str :
        String to allow random orientation or not

    Returns bool :
        True if the vehicle arrived to the end of the travel
    -------

    """
    # return true if arrived
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    x, y, z = state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val
    xf, yf, zf = travel.end_coordinates[0], travel.end_coordinates[1], travel.end_coordinates[2]
    distance_left = np.sqrt((xf - x) ** 2 + (yf - y) ** 2 + (zf - z) ** 2)
    distance_per_t = travel.speed * time_delay
    if orientation == "end":
        yaw = np.arctan2(-(yf - y), xf - x)
        roll = 0
        pitch = 0  # np.arctan2(-(zf-z), np.linalg.norm(distance_left))
    elif orientation == "random":
        yaw = np.radians(random.randint(0, 360))
        roll = 0
        pitch = 0
    else:
        yaw = 0
        roll = 0
        pitch = 0
    if distance_left <= distance_per_t:
        next_x = xf
        next_y = yf
        next_z = zf
        teleport(client, vehicle_name,
                 [next_x, next_y, next_z],
                 [pitch, roll, yaw])
        return True
    else:
        next_x = x + (xf - x) * (travel.speed * time_delay) / distance_left
        next_y = y + (yf - y) * (travel.speed * time_delay) / distance_left
        next_z = z + (zf - z) * (travel.speed * time_delay) / distance_left
        teleport(client, vehicle_name,
                 [next_x, next_y, next_z],
                 [pitch, roll, yaw])
        return False


def create_all_vehicles(client, vehicle_dict):
    """
    Create all the vehicle in the vehicle_dict by calling the API and teleport them below the ground in the "graveyard"
    Parameters
    ----------
    client, MultirotorClient :
        Airsim multirotor client
    vehicle_dict, dict :
        Dictionary containing all the information needed per vehicle

    Returns nothing
    -------

    """
    for key in vehicle_dict.keys():
        create_vehicle(client, vehicle_dict[key]["PawnPath"], vehicle_dict[key]["Name"])
    go_to_graveyard(client, vehicle_dict)


def go_to_graveyard(client, vehicle_dict):  # graveyard slot to vehicle dict ?
    """
    Teleport all the vehicle in the vehicle dict into the "graveyard" (far under the ground) using Airsim API,
    vehicle are teleported at different coordinate to avoid unnecessary collision
    Parameters
    ----------
    client, MultirotorClient :
        Airsim multirotor client
    vehicle_dict, dict :
        Dictionary containing all the information needed per vehicle

    Returns Nothing
    -------

    """
    graveyard_coord = [0, 0, 250]
    time.sleep(1)  # if not the balloon get stuck in the air
    for i in vehicle_dict.keys():
        pose = airsim.Pose(airsim.Vector3r(graveyard_coord[0],
                                           graveyard_coord[1],
                                           graveyard_coord[2]))
        client.simSetVehiclePose(pose, True, vehicle_name=vehicle_dict[i]["Name"])
        client.takeoffAsync(vehicle_dict[i]["Name"])
        # client.moveToPositionAsync(graveyard_coord[0],graveyard_coord[1],graveyard_coord[2]-20,5)
        graveyard_coord[0] += 30


def go_to_start(client, vehicle_list):
    """
    Teleport a list of vehicle to their [0,0,0] coordinates using the Airsim API
    Parameters
    ----------
    client, MultirotorClient :
        Airsim multirotor client
    vehicle_list, list[str] :
        List of vehicle name to be teleported

    Returns Nothing
    -------

    """
    graveyard_coord = [0, 0, 0]
    pose = airsim.Pose(airsim.Vector3r(graveyard_coord[0],
                                       graveyard_coord[1],
                                       graveyard_coord[2]))
    for i in vehicle_list:
        client.simSetVehiclePose(pose, True, vehicle_name=i)


def altitude_type(altitude):
    """
    Randomly affect a new altitude compare to the original one (camera vehicle)
    Parameters
    ----------
    altitude, float :
        Altitude value in NED coordinates system (x,y,-z)

    Returns float :
        New altitude value
    -------

    """
    diff_list = [-100, -50, 0, 50, 100]
    diff = random.choice(diff_list)
    return altitude + diff


def setdetection(client, camera_vehicle, camera_angle, vehicle_dict, detection_radius_m):
    """
    Set the Airsim detection range of specific 3D mesh from vehicle dict
    Parameters
    ----------
    client, MultirotorClient :
        Airsim multirotor client
    camera_vehicle, str :
        Name of the camera vehicle
    camera_angle, list :
        List of each camera angle used
    vehicle_dict, dict :
        Dictionary containing all the information needed per vehicle
    detection_radius_m, float :
        Value in meter for the range of detection of a 3D Mesh by Airsim

    Returns Nothing
    -------

    """
    radius_detect_cm = detection_radius_m * 100
    for cam_name in camera_angle:
        client.simSetDetectionFilterRadius(cam_name, airsim.ImageType.Scene, radius_detect_cm, camera_vehicle)
        for key in vehicle_dict.keys():
            client.simAddDetectionFilterMeshName(cam_name, airsim.ImageType.Scene, vehicle_dict[key]["Mesh"] + "*",
                                                 camera_vehicle)


def find_key_by_name(dictionary, name):
    """
    Find the key of the dictionary containing a specific str (name)
    Parameters
    ----------
    dictionary, dict :
        Dictionary containing the data
    name, str :
        String of the searched name

    Returns Key of the dictionary containing the corresponding Name
    -------

    """
    for key, value in dictionary.items():
        if value.get("Name") == name:
            return key


def check_overlap(bounding_boxes):
    """
    Checks if a list of bounding boxes overlap or not to avoid occlusion issues in training data
    Parameters
    ----------
    bounding_boxes, list :
        List of Box2D bounding boxes

    Returns bool :
        True if overlap else False
    -------

    """
    for i in range(len(bounding_boxes)):
        for j in range(i + 1, len(bounding_boxes)):
            box1 = bounding_boxes[i]
            box2 = bounding_boxes[j]
            if (
                    box1["x_min"] <= box2["x_max"] and
                    box1["x_max"] >= box2["x_min"] and
                    box1["y_min"] <= box2["y_max"] and
                    box1["y_max"] >= box2["y_min"]
            ):
                print("Box overlapping")
                return True
    return False


def checkbb_camera(client, camera_vehicle, camera_angle,
                   vehicle_dict):  # manage if vehicle name different than vehicle mesh
    """
    Check if a flying vehicle was detected in the image and extract the bounding box to save it in a dictionary
    Remove the label on one image if the bounding boxes overlap
    Parameters
    ----------
   client, MultirotorClient :
        Airsim multirotor client
    camera_vehicle, str :
        Name of the camera vehicle
    camera_angle, list :
        List of each camera angle used
    vehicle_dict, dict :
        Dictionary containing all the information needed per vehicle

    Returns label, dict :
        Dictionary containing all the detected vehicle and their data for each images
    -------

    """
    label = {}
    for i in camera_angle:
        info_list = client.simGetDetections(i, airsim.ImageType.Scene, camera_vehicle)
        if not info_list:
            continue
        for info in info_list:
            distance = get_distance([0, 0, 0], [info.relative_pose.position.x_val,
                                                info.relative_pose.position.y_val,
                                                info.relative_pose.position.z_val])
            vehicle_key = find_key_by_name(vehicle_dict, info.name)
            if distance >= vehicle_dict[vehicle_key]["Max_dist_m"]:
                continue
            label[i + "_" + info.name] = {"name": info.name,
                                          "yolo_class": vehicle_dict[vehicle_key]["yolo_class"],
                                          "box2D": {"x_max": info.box2D.max.x_val,
                                                    "y_max": info.box2D.max.y_val,
                                                    "x_min": info.box2D.min.x_val,
                                                    "y_min": info.box2D.min.y_val}}
        if len(info_list) > 1:
            on_same_picture_list = {key: value for key, value in label.items() if key.startswith(i)}
            bbx_on_same_picture = []
            for key in on_same_picture_list.keys():
                bbx_on_same_picture.append(on_same_picture_list[key]["box2D"])
            if check_overlap(bbx_on_same_picture):
                for key in on_same_picture_list.keys():
                    del label[key]
    return label


def get_frame_data(client, camera_vehicle, bbx_dict, timeoftheday, current_level):
    """
    Generate a dictionary of the frame data by concatenating input and requesting data with Airsim API
    Parameters
    ----------
    client, MultirotorClient :
        Airsim multirotor client
    camera_vehicle, str :
        Name of the camera vehicle
    bbx_dict, dict :
        Dictionary containing all the bounding boxes value of the frame
    timeoftheday, str :
        Time of the day of the frame
    current_level, str :
        Current level on the frame

    Returns result, dict :
        Dictionary with all the frame data
    -------

    """
    v1_data = client.getMultirotorState(vehicle_name=camera_vehicle)
    v1_log = {"coordinate":
                  [v1_data.kinematics_estimated.position.x_val,
                   v1_data.kinematics_estimated.position.y_val,
                   v1_data.kinematics_estimated.position.z_val],
              "orientation":
                  [v1_data.kinematics_estimated.orientation.x_val,
                   v1_data.kinematics_estimated.orientation.y_val,
                   v1_data.kinematics_estimated.orientation.z_val]}
    v1_yaw = v1_data.kinematics_estimated.orientation.z_val
    result = {"currentlevel": current_level, "timeoftheday": timeoftheday, camera_vehicle: v1_log}
    name_list = [(details[1]["name"], details[0]) for details in bbx_dict.items()]
    for name in name_list:
        v2_data = client.getMultirotorState(vehicle_name=name[0])
        v2_yaw = v2_data.kinematics_estimated.orientation.z_val
        v2_log = {"coordinate": [v2_data.kinematics_estimated.position.x_val,
                                 v2_data.kinematics_estimated.position.y_val,
                                 v2_data.kinematics_estimated.position.z_val],
                  "orientation": [v2_data.kinematics_estimated.orientation.x_val,
                                  v2_data.kinematics_estimated.orientation.y_val,
                                  v2_data.kinematics_estimated.orientation.z_val]}

        v2_log["distance"] = get_distance(v1_log["coordinate"], v2_log["coordinate"]),
        v2_log["angle"] = np.degrees(v2_yaw - v1_yaw)  # positive diff angle = clockwise
        v2_log["label"] = bbx_dict[name[1]]
        result[name[1]] = v2_log

    return result


def add_to_count(count_dict, bbx_dict):
    """
    Add to the counter dictionary using mutability of dictionary
    Parameters
    ----------
    count_dict, dict :
        Dictionary containing the count of all classes
    bbx_dict, dict :
        Dictionary containing all the bounding boxes value of the frame

    Returns Nothing
    -------

    """
    for key, value in bbx_dict.items():
        class_found = value["yolo_class"]
        count_dict[class_found] += 1


def simulation(client, camera_vehicle, vehicle_dict, camera_angle, save_path: Path, plot, sim_count, current_level,
               timeoftheday_dict, count_dict):  # maybe set other vehicle to a list
    """
    Function to run a whole simulation for each vehicle spawned with random parameters
    Parameters
    ----------
     client, MultirotorClient :
        Airsim multirotor client
    camera_vehicle, str :
        Name of the camera vehicle
    vehicle_dict, dict :
        Dictionary containing all the information needed per vehicle
    camera_angle, list :
        List of each camera angle used
    save_path, Path :
        Path were all the data folder will be stored
    plot, bool :
        Flag to allow plotting or not
    sim_count, int :
        Number of the current simulation
    current_level, str :
        Current level on the frame
    timeoftheday, str :
        Time of the day of the frame
    count_dict, dict :
        Dictionary containing the count of all classes

    Returns log, dict :
        Dictionary containing the data of all frame in the simulation
    -------

    """

    save_path.mkdir(exist_ok=True)
    cam_altitude = random.randint(-700, -400)

    speed = 25
    frame_data = {}
    center_data = client.getMultirotorState(vehicle_name=camera_vehicle)
    center_cam = (center_data.kinematics_estimated.position.x_val,
                  center_data.kinematics_estimated.position.y_val,
                  cam_altitude)

    cam_radius = 1000
    list_travel = []
    cam_travel = generate_travel(center_cam, cam_radius, speed)
    for vehicle in vehicle_dict.keys():
        other_altitude = altitude_type(cam_altitude)
        center_other = (center_data.kinematics_estimated.position.x_val,
                        center_data.kinematics_estimated.position.y_val,
                        other_altitude)
        max_dist = vehicle_dict[vehicle]["Max_dist_m"]
        radius = random.randint(round(max_dist * 0.9), round(max_dist * 1.1))
        list_travel.append(generate_travel(center_other, radius, speed))

    estimate_trajectory(center_cam, cam_radius, cam_travel, list_travel, plot, save_path, sim_count)
    # plot in 3D with height diff
    # set weather
    weather = random.choice([WeatherParameter.Nothing, WeatherParameter.Nothing, WeatherParameter.Rain,
                             WeatherParameter.Snow])
    weather_val = random.randint(20, 80) / 100
    change_weather(client, weather, weather_val)

    # print("wheather : ",weather," | intensity :",weather_val)
    # setup vehicle
    client.enableApiControl(True, vehicle_name=camera_vehicle)
    client.armDisarm(True, vehicle_name=camera_vehicle)
    for vehicle in vehicle_dict.keys():
        client.enableApiControl(True, vehicle_name=vehicle_dict[vehicle]["Name"])
        client.armDisarm(True, vehicle_name=vehicle_dict[vehicle]["Name"])

    other_arrived = False
    cam_arrived = False
    pic_count = 1
    pic_delay = 2  # second now but change to fps next
    # set timeoftheday
    rand_time = random.choice(list(timeoftheday_dict.keys()))
    client.simSetTimeOfDay(True, timeoftheday_dict[rand_time], False, 1, 1000)  # sun not moving for a whole
    teleport(client, camera_vehicle, cam_travel.start_coordinates)
    travel_counter = 0
    for vehicle in vehicle_dict.keys():
        teleport(client, vehicle_dict[vehicle]["Name"], list_travel[travel_counter].start_coordinates)
        travel_counter += 1
    time.sleep(1)
    while other_arrived == False and cam_arrived == False:
        travel_counter = 0
        # print(pic_count)
        # set time of the day
        rand_time = random.choice(list(timeoftheday_dict.keys()))
        client.simSetTimeOfDay(True, timeoftheday_dict[rand_time], False, 1,
                               1000)  # sun not moving for a whole iteration
        cam_arrived = set_next_position(client, cam_travel, camera_vehicle, pic_delay, None)
        for vehicle in vehicle_dict.keys():
            flag_travel = set_next_position(client, list_travel[travel_counter], vehicle_dict[vehicle]["Name"],
                                            pic_delay, "random")
            if flag_travel == True:
                other_arrived = True
            travel_counter += 1
        time.sleep(0.5)
        client.simPause(True)
        bbx_dict = save_img(client, pic_count, camera_angle, save_path, sim_count, camera_vehicle, vehicle_dict)
        add_to_count(count_dict, bbx_dict)
        if bbx_dict:
            frame_data[str(pic_count)] = get_frame_data(client, camera_vehicle, bbx_dict, rand_time, current_level)
            pic_count += 1
        client.simPause(False)
        if pic_count >= 60:
            cam_arrived = True
    time.sleep(2)
    return log_dict(cam_travel, list_travel, weather, weather_val, vehicle_dict, frame_data)


def create_folders(save_path: Path, vehicle_list: list):
    """
    Create folder for each vehicle (not used anymore)
    Parameters
    ----------
    save_path
    vehicle_list

    Returns
    -------

    """
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)
    for i in vehicle_list:
        create = save_path / i
        if not create.exists():
            create.mkdir()


def initialize_level(client, vehicle_dict, camera_angle, camera_vehicle):
    """
    Initialize the level by creating vehicle and setting up the detection range
    Parameters
    ----------
    client, MultirotorClient :
        Airsim multirotor client
    vehicle_dict, dict :
        Dictionary containing all the information needed per vehicle
    camera_angle, list :
        List of each camera angle used
    camera_vehicle, str :
        Name of the camera vehicle

    Returns Nothing
    -------

    """
    # Create vehicles
    create_all_vehicles(client, vehicle_dict)
    detection_radius_m = 1000
    setdetection(client, camera_vehicle, camera_angle, vehicle_dict, detection_radius_m)


def change_level(current_sim, sim_num, level_list):
    """
    Check if the level should change to balance the number of simulation per level
    Parameters
    ----------
    current_sim, int :
        Current simulation number
    sim_num, int :
        Total number of simulation for this dataset
    level_list, list :
        List of level name

    Returns bool :
        True if need to change level
    -------

    """
    if sim_num < len(level_list):
        raise ValueError("Number of simulation smaller than list of level")
    num_level = len(level_list)
    list_change = []
    list_change.append(1)
    for i in range(1, num_level):
        list_change.append(list_change[i - 1] + (sim_num // num_level))

    if current_sim in list_change:
        return True
    else:
        return False


def dataset_generation(save_path: Path, vehicle_dict: dict, vehicle_used, sim_iteration, camera_angle: list, level_list,
                       timeoftheday_dict):
    """
    Main function handling data generation, ordering data, split into train/test/val for YOLOv8 usage
    Be aware that you might need to manually change level on unreal engine if needed
    Parameters
    ----------
    save_path, Path :
        Path were all the data folder will be stored
    vehicle_dict, dict :
        Dictionary containing all the information needed per vehicle :
                    1: {"Name": str,
                    "PawnPath": str,
                    "Mesh": str,
                    "Max_dist_m": float,
                    "yolo_class": str}
    vehicle_used, list :
        List with the name of all the used vehicle
    sim_iteration, int :
        Number of iteration of a simulated travel
    camera_angle, list :
        List of each camera angle used
    level_list, list :
        List of level name
    timeoftheday_dict, dict :
        Dict containing different time of the day, ex :{"time1":"2023-06-15 11:15:00"}

    Returns Nothing
    -------
    If run successfull there should be a train / val / test folder generate with YOLOv8 label format.
    With a good enough balance in the dataset
    """
    # set randomness
    seed_value = random.randint(0, 10000)
    random.seed(seed_value)

    # Create save folder if doesn't exist
    if not (save_path / "data").exists():
        (save_path / "data").mkdir()
    # create_folders(save_path / "data",vehicle_used)

    vehicle_dict = {k: v for k, v in vehicle_dict.items() if any(value in vehicle_used for value in v.values())}

    count_dict = {}
    for key in vehicle_dict.keys():
        count_dict[vehicle_dict[key]["yolo_class"]] = 0

    # Create client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    # Start simulation
    sim_log = {}
    sim_log["random_seed"] = seed_value
    # get setting
    camera_vehicle = "camera_vehicle"
    dict_camera = {}
    for angle in camera_angle:
        test_img = \
        client.simGetImages([airsim.ImageRequest(angle, airsim.ImageType.Scene, False, False)], "camera_vehicle",
                            False)[0]
        dict_camera[angle] = {"resolution": [test_img.width, test_img.height]}
    sim_log["camera_details"] = dict_camera
    sim_count = 1  # create count with index in for
    # Level Management
    level_idx = 0
    current_level = "StartLevel"
    for loop in tqdm(range(1, sim_iteration + 1)):
        if change_level(loop, sim_iteration, level_list):
            current_level = level_list[level_idx]
            time.sleep(2)  # time for engine to load everything
            winsound.PlaySound("mixkit-elevator-tone-2863.wav.wav", winsound.SND_FILENAME)
            input("Change level to " + current_level)
            # client.simLoadLevel(current_level) #Loading Level API is fairly random change manually if needed
            # time.sleep(20) #time for level to load
            # Create client
            client = airsim.MultirotorClient()
            client.confirmConnection()
            client.enableApiControl(True)
            initialize_level(client, vehicle_dict, camera_angle, camera_vehicle)
            level_idx += 1

        sim_log[str(sim_count)] = simulation(client, camera_vehicle, vehicle_dict,
                                             camera_angle, save_path / "data" / "{:02d}".format(sim_count), False,
                                             sim_count, current_level, timeoftheday_dict, count_dict)
        client.reset()  # else the balloon goes flying and generate collisions
        go_to_graveyard(client, vehicle_dict)  # {1:vehicle_dict[other_vehicle]})
        sim_count += 1
    client.simPause(True)
    sim_log["classes_count"] = count_dict
    json_object = json.dumps(sim_log, indent=4)
    with open(str(save_path / "sim_log.json"), "w") as outfile:
        outfile.write(json_object)
    order_dataset(save_path, vehicle_dict)
    generate_train_val_test(save_path / "ordered_data", (80, 10, 10))


def copy_file(source_path: Path, destination_directory: Path, ):
    """
    Copy a file to another folder
    Parameters
    ----------
    source_path, Path :
        Path of the original image
    destination_directory, Path :
        Path of the destination directory

    Returns Nothing
    -------

    """
    filename = source_path.name
    shutil.copy(source_path, destination_directory / filename)
    # time.sleep(0.3) #due to hard drive limitations this sleep makes the copy more fluid


def convert_to_yolo_label(xmax, ymax, xmin, ymin, width, height):
    """
    Convert the bounding boxes from Airsim detection to Yolov8 format :
                        -Xcenter Ycenter width height
    Parameters
    ----------
    xmax
    ymax
    xmin
    ymin
    width
    height

    Returns values for the new bounding box
    -------

    """
    center_x = (xmin + (xmax - xmin) / 2) / width
    center_y = (ymin + (ymax - ymin) / 2) / height
    height_bbx = (ymax - ymin) / height
    width_bbx = (xmax - xmin) / width

    return center_x, center_y, width_bbx, height_bbx


def check_full(xmin, xmax, ymin, ymax, width, height):
    """
    Verify if the object is not touching the border of the image to avoid cropped vehicles
    Parameters
    ----------
    xmin
    xmax
    ymin
    ymax
    width
    height

    Returns bool :
        True if bounding box touches the edge
    -------

    """
    if xmin == 0 or xmax == width or ymin == 0 or ymax == height:
        return False
    else:
        return True


def order_dataset(save_path, vehicle_dict):
    """
    Take the raw data and generate a folder for background and a folder for images & creating label .txt file for YOLOv8
    Parameters
    ----------
    save_path, Path :
        Path to save the new ordered data
    vehicle_dict, dict :
        Dictionary containing all the information needed per vehicle

    Returns Nothing
    -------

    """
    # Create new folders

    new_ordered_path = save_path / "ordered_data"
    if new_ordered_path.exists():
        shutil.rmtree(new_ordered_path)
    new_ordered_path.mkdir()

    list_vehicle = [vehicle_dict[key]["Name"] for key in vehicle_dict]

    if not (new_ordered_path / "img").exists():
        (new_ordered_path / "img").mkdir()

    if not (new_ordered_path / "background").exists():
        (new_ordered_path / "background").mkdir()
    # Load data
    with open(save_path / 'sim_log.json') as json_file:
        sim_log = json.load(json_file)
    resolution_dict = sim_log['camera_details']
    class_count = sim_log["classes_count"]
    del sim_log['random_seed']
    del sim_log['camera_details']
    del sim_log['classes_count']
    sim_log = {int(key): value for key, value in sim_log.items()}

    count_background = 0
    # Order
    for sim_number in tqdm(sim_log.keys()):
        for frame_number, frame_data in tqdm(sim_log[sim_number]['frame_data'].items()):

            list_img_frame = list((save_path / "data" / "{:02d}".format(sim_number)).glob(
                "{:02d}".format(sim_number) + '_' + "{:03d}".format(int(frame_number)) + '_*'))
            list_vehicle_key = list(frame_data.keys())[3:]
            for key in list_vehicle_key:
                vehicle_frame_data = frame_data[key]
                for frame in list_img_frame:
                    angle = key.split("_")[0]
                    if angle in str(frame):
                        xmin = vehicle_frame_data['label']["box2D"]["x_min"]
                        xmax = vehicle_frame_data['label']["box2D"]["x_max"]
                        ymin = vehicle_frame_data['label']["box2D"]["y_min"]
                        ymax = vehicle_frame_data['label']["box2D"]["y_max"]
                        width = resolution_dict[angle]["resolution"][0]
                        height = resolution_dict[angle]["resolution"][1]
                        if check_full(xmin, xmax, ymin, ymax, width, height):  # remove object not fully on the picture
                            copy_file(frame, new_ordered_path / "img")
                            x_center, y_center, width, height = convert_to_yolo_label(xmax, ymax, xmin, ymin, width,
                                                                                      height)
                            with open(new_ordered_path / "img" / (frame.stem + '.txt'), 'a') as file:
                                # remove label if photo is cropped ?
                                file.write(vehicle_frame_data['label']['yolo_class'] + " " +
                                           str(x_center) + " " + str(y_center) + " " +
                                           str(width) + " " + str(height) + "\n")
                    else:
                        count_background += 1
                        if count_background == 3:  # avoid copying all the files
                            count_background = 0
                            copy_file(frame, new_ordered_path / "background")


def create_data_split_folder(folder_name, data_path, dict_class, index_interval):
    """
    Create a data folder with images and label depending on the input with the goal of a balanced dataset
    Parameters
    ----------
    folder_name, str :
        Name of the folder
    data_path, Path :
        Path of the ordered folder data
    dict_class, dict :
        Dictionary with all the images per class
    index_interval, list :
        List of the beginning and ending index for the data in

    Returns Nothing
    -------

    """
    save_path = data_path.parent / folder_name
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)
    (save_path / "images").mkdir()
    (save_path / "labels").mkdir()
    percentage_background = 5
    divided_background = round((100 / len(list(dict_class.keys()))) / percentage_background)
    reduced_background_counter = divided_background - 1
    for key in tqdm(dict_class.keys()):
        for file in range(index_interval[0], index_interval[1] + 1):
            if key == "background":  # no label for background
                reduced_background_counter += 1
                if reduced_background_counter == divided_background:  # reduce the number of background compare to the rest
                    reduced_background_counter = 0
                    copy_file(dict_class[key][file].with_suffix(".png"), save_path / "images")
            else:
                copy_file(dict_class[key][file].with_suffix(".txt"), save_path / "labels")
                copy_file(dict_class[key][file].with_suffix(".png"), save_path / "images")


def generate_train_val_test(data_path: Path, split: tuple):
    """
    Function to generate train / val / test from the ordered data
    Parameters
    ----------
    data_path, Path :
        Path of the ordered folder data
    split, list :
        List of the split for train / val / test, sum must be equal to 100

    Returns
    -------

    """
    if sum(split) != 100:
        raise ValueError("Total of split data isn't equal to 100")
    list_class_folder = data_path.iterdir()
    dict_class = {}

    with open(data_path.parent / 'sim_log.json') as json_file:
        sim_log = json.load(json_file)
    minimum_per_class = min(sim_log["classes_count"].values())
    print(sim_log["classes_count"])
    for key in sim_log["classes_count"].keys():
        dict_class[key] = []

    # create list for each set
    for folder in list_class_folder:
        list_file = list(folder.iterdir())
        list_file = list(set(path.parent / path.stem for path in list_file))
        random.shuffle(list_file)
        dict_class[folder.stem] = list_file

    for img in dict_class['img']:
        with open(str(img) + '.txt') as txt_file:
            lines = txt_file.readlines()
            txt_class_list = []
            for line in lines:
                txt_class = line.strip(" ")[0]
                txt_class_list.append(txt_class)
        dict_class[random.choice(txt_class_list)].append(img)

    for key, value in dict_class.items():
        if len(value) < minimum_per_class:
            minimum_per_class = len(value)

    del dict_class["img"]

    train_interval = [0, round(minimum_per_class * split[0] / 100) - 1]
    val_interval = [round(minimum_per_class * split[0] / 100),
                    round(minimum_per_class * split[0] / 100) + round(minimum_per_class * split[1] / 100) - 1]
    test_interval = [round(minimum_per_class * split[0] / 100) + round(minimum_per_class * split[1] / 100) - 1,
                     minimum_per_class - 1]
    print("Number of label per vehicle :", minimum_per_class)

    create_data_split_folder("val", data_path, dict_class, val_interval)
    create_data_split_folder("train", data_path, dict_class, train_interval)
    create_data_split_folder("test", data_path, dict_class, test_interval)
