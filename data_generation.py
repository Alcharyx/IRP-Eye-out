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
from PIL import Image
import itertools


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
    Nothing = -1
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
        img1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response[0].height, response[0].width, 3)
        gray_image = color.rgb2gray(img_rgb) * 255
        while gray_image.shape == (0,0): #error lag -> empty image
            time.sleep(1)
            #print("Bug empty img")
            response = client.simGetImages([airsim.ImageRequest(i, airsim.ImageType.Scene, False, False)], "", False)
            img1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response[0].height, response[0].width, 3)
            gray_image = color.rgb2gray(img_rgb) * 255

        # print("img ", i)
        img_list.append(gray_image)
    label = checkbb_camera(client, camera_vehicle, camera_angle, vehicle_dict)
    if label:
        for i in range(len(camera_angle)):
            # print("save img ", camera_angle[i])
            frame_path = str(save_path / (
                        "{:02d}".format(sim_count) + '_' + "{:03d}".format(frame) + '_' + camera_angle[i] + '.png '))
            cv2.imwrite(frame_path, img_list[i])
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
    if weather_param !=-1:
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
    a=3

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
    dont_save = False
    time.sleep(1)
    for i in camera_angle:
        info_list = client.simGetDetections(i, airsim.ImageType.Scene, camera_vehicle)
        if not info_list:
            continue
        for info in info_list:
            distance = get_distance([0, 0, 0], [info.relative_pose.position.x_val,
                                                info.relative_pose.position.y_val,
                                                info.relative_pose.position.z_val])
            vehicle_key = find_key_by_name(vehicle_dict, info.name)
            if distance >= vehicle_dict[vehicle_key]["Max_dist_m"]: #and distance <= vehicle_dict[vehicle_key]["Min_dist_m"]: #good for 1 vehicle
                dont_save = True
                print(info.name," is too far")
                continue
            label[i + "_" + info.name] = {"name": info.name,
                                          "yolo_class": vehicle_dict[vehicle_key]["yolo_class"],
                                          "box2D": {"x_max": info.box2D.max.x_val,
                                                    "y_max": info.box2D.max.y_val,
                                                    "x_min": info.box2D.min.x_val,
                                                    "y_min": info.box2D.min.y_val},
                                          "distance": distance}
        if len(info_list) > 1:
            on_same_picture_list = {key: value for key, value in label.items() if key.startswith(i)}
            bbx_on_same_picture = []
            for key in on_same_picture_list.keys():
                bbx_on_same_picture.append(on_same_picture_list[key]["box2D"])
            if check_overlap(bbx_on_same_picture):
                #print("overlap")
                dont_save = True
    if dont_save:
        label ={}
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
    result = {"currentlevel": current_level, "timeoftheday": timeoftheday, camera_vehicle: v1_log}
    name_list = [(details[1]["name"], details[0]) for details in bbx_dict.items()]
    for name in name_list:
        v2_data = client.getMultirotorState(vehicle_name=name[0])
        v2_log = {"coordinate": [v2_data.kinematics_estimated.position.x_val,
                                 v2_data.kinematics_estimated.position.y_val,
                                 v2_data.kinematics_estimated.position.z_val],
                  "orientation": [v2_data.kinematics_estimated.orientation.x_val,
                                  v2_data.kinematics_estimated.orientation.y_val,
                                  v2_data.kinematics_estimated.orientation.z_val]}
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
        class_found = value["yolo_class"] + "_" + value["name"]
        count_dict["individual"][class_found] += 1
        count_dict["group"][value["yolo_class"]] += 1


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
    cam_altitude = random.randint(-800, -500)

    speed = 25
    frame_data = {}
    center_data = client.getMultirotorState(vehicle_name=camera_vehicle)
    center_cam = (center_data.kinematics_estimated.position.x_val,
                  center_data.kinematics_estimated.position.y_val,
                  cam_altitude)

    cam_radius = 1500 #hard coded not good
    list_travel = []
    cam_travel = generate_travel(center_cam, cam_radius, speed)
    for vehicle in vehicle_dict.keys():
        other_altitude = altitude_type(cam_altitude)
        center_other = (center_data.kinematics_estimated.position.x_val,
                        center_data.kinematics_estimated.position.y_val,
                        other_altitude)
        radius_vehicle = vehicle_dict[vehicle]["Max_dist_m"] / 2
        radius = random.randint(round(radius_vehicle * 0.8), round(radius_vehicle * 1.3))
        list_travel.append(generate_travel(center_other, radius, speed))

    #estimate_trajectory(center_cam, cam_radius, cam_travel, list_travel, plot, save_path, sim_count)
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
    pic_delay = 8  # second now but change to fps next
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


        if bbx_dict:
            if len(set([key.split("_")[1] for key in bbx_dict.keys()])) < len(vehicle_dict.keys()):
                #print("anomaly frame :",pic_count)
                count_dict["anomaly"].append(str(save_path / ("{:02d}".format(sim_count) + '_' + "{:03d}".format(pic_count))))
            add_to_count(count_dict, bbx_dict)
            frame_data[str(pic_count)] = get_frame_data(client, camera_vehicle, bbx_dict, rand_time, current_level)
            pic_count += 1

        client.simPause(False)
        if pic_count >= 50:
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


def initialize_level(client, vehicle_dict, camera_angle, camera_vehicle,detection_radius_m):
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
                       timeoftheday_dict, distance_interval):
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
    distance_interval, list:
        Interval that changes the Max_dist_m and Min_dist_m in dict_vehicle if the current value is outside of the
        interval

    Returns Nothing
    -------
    If run successfull there should be a train / val / test folder generate with YOLOv8 label format.
    With a good enough balance in the dataset
    """
    # set randomness
    seed_value = random.randint(0, 10000)
    random.seed(seed_value)

    # Create save folder if doesn't exist
    if (save_path / "data").exists():
        shutil.rmtree(save_path / "data")
    (save_path / "data").mkdir()
    # create_folders(save_path / "data",vehicle_used)

    vehicle_dict = {k: v for k, v in vehicle_dict.items() if any(value in vehicle_used for value in v.values())}
    for key in vehicle_dict.keys():
        if vehicle_dict[key]["Min_dist_m"] < distance_interval[0]:
            vehicle_dict[key]["Min_dist_m"] = distance_interval[0]

        if vehicle_dict[key]["Max_dist_m"] > distance_interval[1]:
            vehicle_dict[key]["Max_dist_m"] = distance_interval[1]

    count_dict = {"individual":{},
                  "group":{},
                  "anomaly": []}
    for key in vehicle_dict.keys():
        count_dict["individual"][vehicle_dict[key]["yolo_class"] + "_" + vehicle_dict[key]["Name"]] = 0
        count_dict["group"][vehicle_dict[key]["yolo_class"]] = 0

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
            initialize_level(client, vehicle_dict, camera_angle, camera_vehicle,10000)
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

def remove_from_right(string, character):
    index = string.rfind(character)
    if index != -1:
        return string[:index]
    else:
        return string



def order_dataset(save_path, distance_interval):
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
    temp_counter = 0
    temp_counter_miss = 0
    # Create new folders

    # Load data
    with open(save_path / 'sim_log.json') as json_file:
        sim_log = json.load(json_file)
    result_dict = {"background": [],
                   "img": [],
                   "classes_count": {"individual" : sim_log["classes_count"]["individual"],
                                     "group" : sim_log["classes_count"]["group"]}}
    resolution_dict = sim_log['camera_details']
    del sim_log['random_seed']
    del sim_log['camera_details']
    del sim_log['classes_count']
    sim_log = {int(key): value for key, value in sim_log.items()}
    count_count = 0
    for key in sim_log.keys():
        for frame_data in sim_log[key]["frame_data"]:
            count_count += (len(sim_log[key]["frame_data"][frame_data]) -3)
    count_background = 0
    # Order
    for sim_number in tqdm(sim_log.keys()):
        for frame_number, frame_data in sim_log[sim_number]['frame_data'].items():

            list_img_frame = list((save_path / "data" / "{:02d}".format(sim_number)).glob(
                "{:02d}".format(sim_number) + '_' + "{:03d}".format(int(frame_number)) + '_*'))
            angle_list = []
            suffix_list = []
            for i in range(len(list_img_frame)):
                suffix_list.append(list_img_frame[i].suffix)
                angle_list.append(list_img_frame[i].stem.split("_")[2])
                list_img_frame[i] = list_img_frame[i].parent / remove_from_right(list_img_frame[i].stem,"_")
            #counter_label = 0
            for index_angle in range(len(angle_list)):
                angle = angle_list[index_angle]
                key_to_verify = []
                for key in list(frame_data.keys())[3:]:
                    on_angle = key.split("_")[0]
                    if on_angle == angle:
                        key_to_verify.append(key)
                flag_to_save = True
                if not key_to_verify:
                    count_background += 1
                    if count_background == 3:  # avoid copying all the files
                        count_background = 0
                        result_dict["background"].append([str(list_img_frame[index_angle].with_name(list_img_frame[index_angle].name + "_" + angle + suffix_list[index_angle])),[]])
                        #copy_file(list_img_frame[index_angle].with_name(list_img_frame[index_angle].name + "_" + angle + suffix_list[index_angle]),
                        #          new_ordered_path / "background")
                    continue
                label_to_save = {}
                counter_label = 0
                for key in key_to_verify:
                    if not flag_to_save:
                        continue
                    xmin = frame_data[key]['label']["box2D"]["x_min"]
                    xmax = frame_data[key]['label']["box2D"]["x_max"]
                    ymin = frame_data[key]['label']["box2D"]["y_min"]
                    ymax = frame_data[key]['label']["box2D"]["y_max"]
                    width = resolution_dict[angle]["resolution"][0]
                    height = resolution_dict[angle]["resolution"][1]
                    distance = frame_data[key]['label']["distance"]
                    if not check_full(xmin, xmax, ymin, ymax, width, height) or distance < distance_interval[0] or distance > distance_interval[1]: # remove object not fully on the picture be careful if sim bug can happen a lot
                        flag_to_save = False
                        temp_counter_miss += len(key_to_verify)
                        for i in key_to_verify:
                            for j in result_dict["classes_count"]["individual"].keys():
                                if j.split("_")[1] in i:
                                    result_dict["classes_count"]["individual"][j] -= 1
                                    result_dict["classes_count"]["group"][j.split("_")[0]] -= 1
                        continue
                    x_center, y_center, width, height = convert_to_yolo_label(xmax, ymax, xmin, ymin, width,
                                                                              height)
                    label_to_save[counter_label] = frame_data[key]['label']['yolo_class'] +\
                                                   " " + str(x_center) + \
                                                   " " + str(y_center) + \
                                                   " " + str(width) + \
                                                   " " + str(height) + "\n"
                    counter_label += 1
                if flag_to_save:
                    #copy_file(list_img_frame[index_angle].with_name(
                    #    list_img_frame[index_angle].name + "_" + angle + suffix_list[index_angle]),
                    #          new_ordered_path / "img")
                    label =[]
                    for key in range(len(label_to_save.keys())):
                        temp_counter +=1
                        #with open(new_ordered_path / "img" / (list_img_frame[index_angle].name + "_" + angle + '.txt'), 'a') as file:
                        #    file.write(label_to_save[key])
                        label.append(label_to_save[key])
                    result_dict["img"].append([str(list_img_frame[index_angle].with_name(list_img_frame[index_angle].name + "_" + angle + suffix_list[index_angle])),
                                               label])
    print(temp_counter)
    print(temp_counter_miss)
    json_object = json.dumps(result_dict, indent=4)
    with open(str(save_path / ("ordered_data_"+ str(distance_interval) +".json")), "w") as outfile:
        outfile.write(json_object)


def create_data_split_folder(folder_path, list_file):
    """
    Create a data folder with images and label depending on the input
    Parameters
    ----------
    folder_path, str :
        Name of the folder
    list_file, list[Path] :
        List of all the path to save in the folder

    Returns Nothing
    -------

    """
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True)
    (folder_path / "images").mkdir()
    (folder_path / "labels").mkdir()

    for file in tqdm(list_file):
        if file.suffix == ".txt":
            copy_file(file, folder_path / "labels")
        else:
            copy_file(file,folder_path / "images")



def generate_combinations(list):
    combinations = []
    for index in range(1, len(list) + 1):
        for combination in itertools.combinations(list, index):
            fused_combination = '_'.join(elem.split("_")[0] for elem in combination)
            combinations.append(fused_combination)
    return combinations


def generate_train_val_test(ordered_json_path: Path, split: tuple, name: str,rand_seed):
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
    random.seed(rand_seed)
    if sum(split) != 100:
        raise ValueError("Total of split data isn't equal to 100")
    with open(ordered_json_path) as json_file:
        ordered_log = json.load(json_file)
    classes_count = ordered_log["classes_count"]
    del ordered_log["classes_count"]
    dict_split = {"train": [],
                  "val": [],
                  "test": []}
    label_path = (ordered_json_path.parent / "labels")
    if label_path.exists():
        shutil.rmtree(label_path)
    label_path.mkdir(parents=True)

    with open(ordered_json_path.parent / 'sim_log.json') as json_file:
        sim_log = json.load(json_file)
    maximum_per_class = min(sim_log["classes_count"]["group"].values())
    dict_number_vehicle_per_class ={}
    for i in sim_log["classes_count"]["group"].keys():
        dict_number_vehicle_per_class[i] = 0
        for key in sim_log["classes_count"]["individual"].keys():
            if i == key.split("_")[0]:
                dict_number_vehicle_per_class[i] += 1

    #print(sim_log["classes_count"])
    percent_background = 5
    background_files = []
    dict_class = {}
    for i in generate_combinations(sorted(list(classes_count["individual"].keys()),reverse=True)):
        dict_class[i] = 0
        dict_class["path_" + i] = []
    for key_img_type in ordered_log.keys():
        if key_img_type == "background":
            background_files = []
            for i in ordered_log[key_img_type]:
                background_files.append(Path(i[0]))
            random.shuffle(background_files)
            continue #manage background after to add 5% of it
        list_file = ordered_log[key_img_type]
        random.shuffle(list_file)

        for img_data in list_file:
            img_path = img_data[0]
            label = img_data[1]
            with open(str(Path(label_path / Path(img_path).stem).with_suffix(".txt")),"a") as txt_file:
                txt_file.writelines(label)
            txt_class_list=[]
            for i in label:
                txt_class_list.append(i.split(" ")[0])
            txt_class_list = sorted(txt_class_list,reverse=True)
            key = '_'.join(str(elem) for elem in txt_class_list)
            dict_class["path_"+key].append(img_path)
            dict_class[key] += 1
    excess_dict = {}
    for key in classes_count["group"]:
        excess_dict[key] = classes_count["group"][key] - maximum_per_class
    stuck_counter = 0
    while sum(excess_dict.values()) > 0:
        virtual_dict = {}
        key_to_eliminate = ""
        for key in excess_dict.keys():
            virtual_excess = excess_dict[key]
            if excess_dict[key] > 0:
                while key_to_eliminate.count(key) < dict_number_vehicle_per_class[key] and virtual_excess!=0:
                    key_to_eliminate = key_to_eliminate + key + "_"
                    virtual_excess -= 1
            virtual_dict[key] = virtual_excess

        stuck_counter += 1
        if stuck_counter >= 30: #in case it cannot find a possible classes combination to remove
            excess_dict = virtual_dict
            print("bad balance")
        while key_to_eliminate != "":
            if len(dict_class["path_"+key_to_eliminate[:-1]]) > 3:  #to keep one sample of each type could be increased
                key_to_eliminate = key_to_eliminate[:-1]
                del dict_class["path_"+ key_to_eliminate][0]
                dict_class[key_to_eliminate] -=1
                excess_dict = virtual_dict
                key_to_eliminate =""
            else:
                index_to_eliminate = random.choice(np.linspace(0,len(key_to_eliminate)-2,(len(key_to_eliminate))//2))
                key_to_eliminate = remove_characters_by_indices(key_to_eliminate,[int(index_to_eliminate),int(index_to_eliminate)+1])


    for key in [key for key in dict_class.keys() if "path" not in key]:
        amount_test = round(dict_class[key] * split[2]/100)
        amount_val = round(dict_class[key] * split[1]/100)
        amount_train = dict_class[key] - amount_val - amount_test
        for i in range(amount_test):
            current_file = Path(dict_class["path_"+key].pop(0))
            dict_split["test"].append(current_file)
            dict_split["test"].append((label_path / current_file.stem).with_suffix(".txt"))
        for i in range(amount_test):
            current_file = Path(dict_class["path_" + key].pop(0))
            dict_split["val"].append(current_file)
            dict_split["val"].append((label_path / current_file.stem).with_suffix(".txt"))
        for i in range(amount_train):
            current_file = Path(dict_class["path_" + key].pop(0))
            dict_split["train"].append(current_file)
            dict_split["train"].append((label_path / current_file.stem).with_suffix(".txt"))

    if background_files:
        for key in dict_split.keys():
            number_background = math.ceil(len(dict_split[key]) * percent_background / 100)
            for i in range(number_background):
                if background_files: #in case not enough background
                    dict_split[key].append(background_files.pop(0))
                else:
                    print("generate more background in ordered data")
            random.shuffle(dict_split[key])

    create_data_split_folder(ordered_json_path.parent / ("val" + name), dict_split["val"])
    create_data_split_folder(ordered_json_path.parent / ("test" + name), dict_split["test"])
    create_data_split_folder(ordered_json_path.parent / ("train" + name), dict_split["train"])

def remove_characters_by_indices(string, indices):
    new_string = ""
    for i in range(len(string)):
        if i not in indices:
            new_string += string[i]
    return new_string

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
        return 1,1
    a = int(number ** 0.5)
    while a > 0:
        if number % a == 0:
            b = number // a
            if a == 1:
                raise ValueError("The number of slice cannot be divided properly change the number of slice")
            else:
                return a, b
        a -= 1

def is_label_on_image(width_pixels,height_pixels,label_dict,slice_number):
    """
    Check if the bounding box is on the slice of the image
    Parameters
    ----------
    width_pixels, list :
        Interval of pixel on x axis for the image
    height_pixels, list :
        Interval of pixel on y axis for the image
    label_dict, dict :
        Dictionary containing the label bounding box pixel coordinate
    slice_number, list :
        Contain the slice number [row, column]

    Returns result, list :
        Flag if the slice is usable or not Dictionary with the new coordinate for the label of the slice
    -------

    """
    result = {str(slice_number) : {"State": True}}
    dict_in = {}
    to_del = []

    for key in label_dict.keys():
        xcenter, ycenter, xsize, ysize = label_dict[key]
        bb_xmin = xcenter - xsize/2
        bb_ymin = ycenter - ysize/2
        bb_xmax = xcenter + xsize/2
        bb_ymax = ycenter + ysize/2
        if (bb_xmin > width_pixels[0] and bb_xmax < width_pixels[1]
            and bb_ymin > height_pixels[0] and bb_ymax < height_pixels[1]):
                dict_in[key] = "Inside"
        elif ((bb_xmin < width_pixels[0] and bb_xmax < width_pixels[0]) or
              (bb_xmin > width_pixels[1] and bb_xmax > width_pixels[1]) or
              (bb_ymin < height_pixels[0] and bb_ymax < height_pixels[0]) or
              (bb_ymin > height_pixels[1] and bb_ymax > height_pixels[1])):
                dict_in[key] = "Outside"
                #Outside
        else:
            dict_in[key] = "Partial"
            result[str(slice_number)]["State"] = False
            #print("Partial")
            #Partial
    for key in dict_in.keys():
        if dict_in[key] == "Inside":
            result[str(slice_number)][key] = new_coordinates(width_pixels, height_pixels, label_dict[key])
            to_del.append(key)
    #for i in to_del:
    #    del label_dict[i]
    return result

def new_coordinates(width_pixels,height_pixels,original_label):
    width = width_pixels[1] - width_pixels[0]
    height = height_pixels[1] - height_pixels[0]
    original_label[0] -= width_pixels[0]
    original_label[1] -= height_pixels[0]
    return convert_to_yolo_label(original_label[0] + original_label[2]/2,original_label[1] + original_label[3]/2,
                                 original_label[0] - original_label[2]/2,original_label[1] - original_label[3]/2,
                                 width,height)

def order_slice_data(order_path,number_of_slice, overlap,distance_interval):
    temp_counter = [0]
    save_path = order_path.parent / ("ordered_slice_data_" + str(distance_interval))
    data_slice_path = save_path.parent / "data_slice"
    if data_slice_path.exists():
        shutil.rmtree(data_slice_path)
    data_slice_path.mkdir()

    with open(order_path) as json_file:
        order_data = json.load(json_file)
    slice_ordered_dict = {"classes_count": {"individual" : order_data["classes_count"]["individual"],
                                     "group" : order_data["classes_count"]["group"]}}
    for i in slice_ordered_dict["classes_count"]["group"].keys():
        slice_ordered_dict["classes_count"]["group"][i] =0
    del order_data["classes_count"]

    for key in order_data.keys():
        slice_ordered_dict[key] = []
    for key in order_data.keys():
        for img in tqdm(order_data[key]):
            slice_img(img,data_slice_path, slice_ordered_dict, number_of_slice, overlap,temp_counter)
    print(temp_counter[0])
    json_object = json.dumps(slice_ordered_dict, indent=4)
    with open(str(save_path.parent / ("ordered_data_slice_"+ str(distance_interval) +".json")), "w") as outfile:
        outfile.write(json_object)


def slice_img(img, data_slice_path, slice_ordered_data_dict, number_of_slice, overlap,temp_counter):
    """
    Slice a Grayscale image into a number of slices with a defined overlap
    Parameters
    ----------
    img_path, Path :
        Path of the image to be sliced
    label, str :
        string for the label
    number_of_slice, int :
        Total number of slices for the image
    overlap, float :
        Value of the overlap between slices 20% = 0.2

    Returns slices, list :
        List of all the sliced images
    -------

    """
    label = img[1]
    img_path = Path(img[0])
    image = Image.open(str(img_path))
    image_array = np.array(image)
    height, width = image_array.shape
    num_width, num_height = good_split_slice(number_of_slice)
    slice_width = width // num_width
    slice_height = height // num_height
    overlap_width = round(slice_width * overlap)
    overlap_height = round(slice_height * overlap)
    slices = []
    label_dict= {}
    new_ordered_data_dict = {"background": []}
    flag_background = True
    if label:
        new_ordered_data_dict[img_path.parents[0]] = []
        flag_background = False
        counter = 1
        for line in label:
            line_data = line.strip("\n").split(" ")
            line_data = [float(data) for data in line_data]
            label_dict[str(int(line_data[0]))+"_"+str(counter)] = [round(line_data[1] * width),
                                                round(line_data[2] * height),
                                                math.ceil(line_data[3] * width),
                                                math.ceil(line_data[4] * height)]
            counter += 1
    new_coordinates = {}
    for i in range(num_height):
        start_height = i * (slice_height - round(overlap_height/2))
        end_height = start_height + slice_height + round(overlap_height/2)
        if end_height > height: #don't go over the img, to keep the slice of same size
            diff = end_height - height
            end_height = height
            start_height-= diff
        for j in range(num_width):
            start_width = j * (slice_width - round(overlap_width/2))
            end_width = start_width + slice_width + round(overlap_width/2)
            if end_width > width:
                diff = end_width - width
                end_width = width
                start_width -= diff

            slice_img = image_array[start_height:end_height, start_width:end_width]
            if flag_background:
                slices.append(slice_img)
            else:
                new_coordinates.update(is_label_on_image([start_width,end_width],[start_height,end_height],label_dict,[i,j]))
                #add slice if inside image
                slices.append(slice_img)
                #img_slice = Image.fromarray(slice_img)
                # save img
                #img_slice.show()
                #img_slice.save("slide"+str(i)+"_"+str(j)+".png")
    if flag_background:
        for column in range(num_width):
            for row in range(num_height):
                if random.choice([True,False, False, False]):
                    img_slice = Image.fromarray(slices[column * num_width + row])
                    img_slice.save(str(data_slice_path / (img_path.stem + "_" + str(column) + "_" + str(row) + ".png")))
                    slice_ordered_data_dict["background"].append([str(data_slice_path / (img_path.stem + "_" + str(column) + "_" + str(row) + ".png")),[]])
                    #with open(save_path / (img_path.stem + "_" + str(column) + "_" + str(row)+ ".txt"), 'a') as file:
                    #    file.write("")
    else:
        to_del = []
        for key in new_coordinates.keys():
            if new_coordinates[key]["State"] == False:
                to_del.append(key)
        for i in to_del:
            del new_coordinates[i]
        no_duplicate = False
        while no_duplicate == False:
            no_duplicate = remove_dupli_recu(new_coordinates)
        if not new_coordinates:
            print(str(img_path))
            print("not normal")
        for key in new_coordinates.keys():
            column,row = [int(item.strip()) for item in key[1:-1].split(',')]
            img_slice = Image.fromarray(slices[column * num_width + row])

            if not list(new_coordinates[key].keys())[1:]:
                if random.choice([True,False, False, False]):
                    img_slice.save(str(data_slice_path / (img_path.stem + "_" + str(column) + "_" + str(row) + ".png")))
                    slice_ordered_data_dict["background"].append(
                        [str(data_slice_path / (img_path.stem + "_" + str(column) + "_" + str(row) + ".png")), []])
                continue

            img_slice.save(str(data_slice_path / (img_path.stem + "_" + str(column) + "_" + str(row) + ".png")))
            new_label = []
            for label_key in list(new_coordinates[key].keys())[1:]:
                temp_counter[0] += 1
                current_label = label_key.split("_")[0]
                for i in slice_ordered_data_dict["classes_count"]["group"].keys():
                    if i == current_label:
                        slice_ordered_data_dict["classes_count"]["group"][current_label] += 1

                new_label.append(current_label + " " +
                               str(new_coordinates[key][label_key][0]) + " " + str(new_coordinates[key][label_key][1]) + " " +
                               str(new_coordinates[key][label_key][2]) + " " + str(new_coordinates[key][label_key][3]) + "\n")
                with open(data_slice_path / (img_path.stem + "_" + str(column) + "_" + str(row)+ ".txt"), 'a') as file:
                    file.write(label_key.split("_")[0] + " " +
                               str(new_coordinates[key][label_key][0]) + " " + str(new_coordinates[key][label_key][1]) + " " +
                               str(new_coordinates[key][label_key][2]) + " " + str(new_coordinates[key][label_key][3]) + "\n")
            slice_ordered_data_dict["img"].append([str(data_slice_path / (img_path.stem + "_" + str(column) + "_" + str(row) + ".png")),
                                                              new_label])

def remove_dupli_recu(new_coordinates):
    common_label_keys = set()
    key_with_common =set()
    for key_i in list(new_coordinates.keys()):
        for key_j in list(new_coordinates.keys()):
            if key_j == key_i:
                continue
            current_common = set(new_coordinates[key_i].keys()).intersection(set(new_coordinates[key_j].keys()))
            current_common.remove('State')
            for i in current_common:
                common_label_keys.add(i)
            if current_common:
                key_with_common.add(key_j)
                key_with_common.add(key_i)

    if key_with_common:
        best_len = -1
        best_key_to_remove = -1

        for removed_key in key_with_common:

            current_data = []
            compile = [main_key for main_key in new_coordinates.keys() if main_key != removed_key]
            for current_key in compile:
                for label_key in new_coordinates[current_key].keys():
                    current_data.append(label_key)
            current_data = set(current_data)
            current_data.remove('State')
            if len(current_data) == best_len: #random chance for change so that it's not alway the same that gets remove
                                              # and more diversity in the position of the labels that are mainly on
                                              # the side of images for split
                if random.choice([1,2]) == 1:
                    continue
                else:
                    best_len = len(current_data)
                    best_key_to_remove = removed_key
            elif len(current_data) > best_len:
                best_len = len(current_data)
                best_key_to_remove = removed_key

        del new_coordinates[best_key_to_remove]
        return False
    return True


