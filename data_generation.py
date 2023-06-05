import math
import time
from pathlib import Path
import shutil
import random
import numpy as np
import airsim
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json


class travel:
    def __init__(self,start,end,speed,angle_start,angle_end):
        self.start_coordinates = start
        self.end_coordinates = end
        self.speed = speed
        self.angle_start = angle_start
        self.angle_end = angle_end

class WeatherParameter:
    Nothing = None
    Rain = 0
    #Roadwetness = 1
    Snow = 2
    #RoadSnow = 3
    #MapleLeaf = 4
    #RoadLeaf = 5
    Dust = 6
    Fog = 7 #very low visibility


def save_img(client, frame,camera_angle, save_path,sim_count):
    """
    Save images from different angles with high resolution camera
    Parameters
    ----------
    client
    counter
    save_path

    Returns
    -------

    """
    img_list = []
    for i in camera_angle:
        response = client.simGetImages([airsim.ImageRequest(i, airsim.ImageType.Scene, False, False)])
        #print("img ", i)
        img_list.append(response[0])
    for i in range(len(camera_angle)):
        #print("save img ", camera_angle[i])
        img1d = np.frombuffer(img_list[i].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(img_list[i].height, img_list[i].width, 3)
        airsim.write_png(str(save_path / ("{:02d}".format(sim_count) + '_' + "{:03d}".format(frame) + '_' + camera_angle[i] + '.png ')), img_rgb)


def create_vehicle(client, pawn_path,name):
    """
    Create a vehicle with a special 3D mesh
    Parameters
    ----------
    client
    pawn_path
    coordinates

    Returns
    -------

    """
    pose = airsim.Pose(airsim.Vector3r(0,0,-2))
    client.simAddVehicle(name, "simpleflight", pose, pawn_path)
    client.enableApiControl(True, vehicle_name=name)


def teleport(client, vehicle_name, coordinates,orientation = [0,0,0]):
    """
    Teleport a vehicle to coordinates and hover
    Parameters
    ----------
    client
    vehicle_name
    coordinates

    Returns
    -------

    """
    pose = airsim.Pose(airsim.Vector3r(coordinates[0],
                                        coordinates[1],
                                        coordinates[2]),
                        airsim.to_quaternion(orientation[0],
                                             orientation[1],
                                             orientation[2]))
    client.simSetVehiclePose(pose, True, vehicle_name)
    #client.moveToZAsync(coordinates[2], 10,vehicle_name=vehicle_name).join()
    client.takeoffAsync(vehicle_name=vehicle_name).join()
    #client.hoverAsync(vehicle_name).join()


def get_coordinates(center,radius, angle):
    x = center[0] + radius * math.cos(math.radians(angle))
    y = center[1] + radius * math.sin(math.radians(angle))
    coordinates = [x,y,center[2]]
    return coordinates


def get_distance(coord1,coord2):
    return math.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)


def generate_travel(center,radius,speed):
    flag_farenough = False
    angle_start = random.randint(1,360)
    start = get_coordinates(center,radius,angle_start)
    while not flag_farenough:
        angle_end = random.randint(0,360)
        end = get_coordinates(center,radius,angle_end)
        distance = get_distance(start,end)
        #print(distance)
        if distance >= radius*1.5:
            flag_farenough = True

    altitude_change = random.uniform(-1,1) * center[2] * random.randint(5,20) / 100
    start[2] = start[2] + round(altitude_change/2)
    end[2] = end[2] - round(altitude_change/2)
    trip = travel(start,end,speed,angle_start,angle_end)

    return trip


def estimate_trajectory(center,radius,travel1: travel,travel2: travel,plot:bool,save_path:Path,sim_count):
    fig =plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    #Travel1
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = min(travel1.end_coordinates[2],travel1.start_coordinates[2]) * np.ones_like(theta)
    ax.plot(x, y, z, color='green')

    ax.quiver(
        travel1.start_coordinates[0],travel1.start_coordinates[1],travel1.start_coordinates[2],
        travel1.end_coordinates[0]-travel1.start_coordinates[0],
        travel1.end_coordinates[1]-travel1.start_coordinates[1],
        travel1.end_coordinates[2]-travel1.start_coordinates[2],
        arrow_length_ratio= 0.1, color='red')

    #Travel2
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = max(travel2.end_coordinates[2], travel2.start_coordinates[2]) * np.ones_like(theta)
    ax.plot(x, y, z, color='green')
    ax.quiver(
        travel2.start_coordinates[0], travel2.start_coordinates[1], travel2.start_coordinates[2],
        travel2.end_coordinates[0] - travel2.start_coordinates[0],
        travel2.end_coordinates[1] - travel2.start_coordinates[1],
        travel2.end_coordinates[2] - travel2.start_coordinates[2],
        arrow_length_ratio=0.1, color='blue')
    plt.savefig(save_path / ("{:02d}".format(sim_count) +"_estimated_path.png"))
    if plot:
        plt.show()

def log_dict(cam_travel,other_travel, weather_param, weather_val,center_cam,center_other,radius, other_vehicle_dict,frame_data):
    log = {}
    log["other_vehicle"] = other_vehicle_dict
    log["circle_cam"] = {"center": center_cam,
                     "radius": radius}
    log["circle_vehicle"] = {"center": center_other,
                         "radius": radius}
    log["travel_cam"] = {"start": cam_travel.start_coordinates,
                         "end": cam_travel.end_coordinates,
                         "speed":cam_travel.speed,
                         "angle_start": cam_travel.angle_start,
                         "angle_end": cam_travel.angle_end
                         }
    log["travel_other"] = {"start": other_travel.start_coordinates,
                         "end": other_travel.end_coordinates,
                         "speed":other_travel.speed,
                         "angle_start": other_travel.angle_start,
                         "angle_end": other_travel.angle_end
                         }
    log["weather"] = {"param": weather_param,
                      "val": weather_val}
    log["frame_data"] = frame_data
    return log

def change_weather(client: airsim.MultirotorClient, weather_param: WeatherParameter,val):
    if weather_param is not None:
        client.simEnableWeather(True)
        client.simSetWeatherParameter(weather_param,val)


def set_next_position(client,travel:travel,vehicle_name,t,orientation):
    #return true if arrived
    state = client.getMultirotorState(vehicle_name=vehicle_name)
    x,y,z = state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val
    xf,yf,zf = travel.end_coordinates[0],travel.end_coordinates[1],travel.end_coordinates[2]
    distance_left = np.sqrt((xf - x) ** 2 + (yf - y) ** 2 + (zf - z) ** 2)
    distance_per_t = travel.speed * t
    if orientation == "end":
        yaw = np.arctan2(-(yf - y),xf- x)
        roll = 0
        pitch = 0#np.arctan2(-(zf-z), np.linalg.norm(distance_left))
    elif orientation == "random":
        yaw = np.radians(random.randint(0,360))
        roll = 0
        pitch = np.radians(random.randint(-50,50))
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
        next_x = x + (xf - x) * (travel.speed * t) / distance_left
        next_y = y + (yf - y) * (travel.speed * t) / distance_left
        next_z = z + (zf - z) * (travel.speed * t) / distance_left
        teleport(client, vehicle_name,
                 [next_x, next_y, next_z],
                 [pitch, roll, yaw])
        return  False


def create_all_vehicles(client,vehicle_dict):
    for key in vehicle_dict.keys():
        create_vehicle(client, vehicle_dict[key]["PawnBP"], vehicle_dict[key]["Name"])
    go_to_graveyard(client,vehicle_dict)


def go_to_graveyard(client,vehicle_dict): #graveyard slot to vehicle dict ?
    graveyard_coord = [0,0,250]
    time.sleep(1) #if not the balloon get stuck in the air
    for i in vehicle_dict.keys():
        pose = airsim.Pose(airsim.Vector3r(graveyard_coord[0],
                                           graveyard_coord[1],
                                           graveyard_coord[2]))
        client.simSetVehiclePose(pose,True,vehicle_name=vehicle_dict[i]["Name"])
        client.takeoffAsync(vehicle_dict[i]["Name"])
        #client.moveToPositionAsync(graveyard_coord[0],graveyard_coord[1],graveyard_coord[2]-20,5)
        graveyard_coord[0] += 30


def go_to_start(client,vehicle_list):
    graveyard_coord = [0, 0, 0]
    pose = airsim.Pose(airsim.Vector3r(graveyard_coord[0],
                                       graveyard_coord[1],
                                       graveyard_coord[2]))
    for i in vehicle_list:
        client.simSetVehiclePose(pose,True,vehicle_name=i)


def altitude_type():
    diff_list = [-50,0,50]
    diff = random.choice(diff_list)
    altitude = random.randint(-500, -300)
    return altitude, altitude+diff

def setdetection(client, camera_vehicle, camera_angle,vehicle_dict,detection_radius_m):
    radius_detect_cm = detection_radius_m * 100
    for cam_name in camera_angle:
        client.simSetDetectionFilterRadius(cam_name, airsim.ImageType.Scene, radius_detect_cm, camera_vehicle)
        for key in vehicle_dict.keys():
            client.simAddDetectionFilterMeshName(cam_name, airsim.ImageType.Scene, vehicle_dict[key]["Mesh"] + "*", camera_vehicle)


def checkbb_camera(client,camera_vehicle,camera_angle,other_vehicle_dict): #manage if vehicle name different than vehicle mesh
    label = {}
    for i in camera_angle:
        info_list = client.simGetDetections(i,airsim.ImageType.Scene,camera_vehicle)
        if not info_list:
            continue
        for info in info_list:
            if info.name != other_vehicle_dict["Name"]:
                continue
            distance = get_distance([0,0,0],[info.relative_pose.position.x_val,
                                             info.relative_pose.position.y_val,
                                             info.relative_pose.position.z_val])
            if distance >= other_vehicle_dict["Max_dist_m"]:
                continue
            label[i] = {"name": info.name,
                        "box2D": {"x_max": info.box2D.max.x_val,
                                  "y_max": info.box2D.max.y_val,
                                  "x_min": info.box2D.min.x_val,
                                  "y_min": info.box2D.min.y_val}}
    return label


def get_frame_data(client,camera_vehicle, other_vehicle_dict,camera_angle):
    v1_data = client.getMultirotorState(vehicle_name=camera_vehicle)
    v2_data = client.getMultirotorState(vehicle_name=other_vehicle_dict["Name"])
    v1_yaw = v1_data.kinematics_estimated.orientation.z_val
    v2_yaw = v2_data.kinematics_estimated.orientation.z_val
    v1_log = {"coordinate":
                  [v1_data.kinematics_estimated.position.x_val,
                   v1_data.kinematics_estimated.position.y_val,
                   v1_data.kinematics_estimated.position.z_val],
              "orientation":
                  [v1_data.kinematics_estimated.orientation.x_val,
                   v1_data.kinematics_estimated.orientation.y_val,
                   v1_data.kinematics_estimated.orientation.z_val]}
    v2_log = {"coordinate":
                  [v2_data.kinematics_estimated.position.x_val,
                   v2_data.kinematics_estimated.position.y_val,
                   v2_data.kinematics_estimated.position.z_val],
              "orientation":
                   [v2_data.kinematics_estimated.orientation.x_val,
                    v2_data.kinematics_estimated.orientation.y_val,
                    v2_data.kinematics_estimated.orientation.z_val]}
    dist = get_distance(v1_log["coordinate"],v2_log["coordinate"])
    label = checkbb_camera(client, camera_vehicle, camera_angle,other_vehicle_dict)

    return {camera_vehicle:v1_log, other_vehicle_dict["Name"]:v2_log, "distance":dist, "angle":np.degrees(v2_yaw - v1_yaw),"label":label} #positive diff = clockwise



def simulation(client,camera_vehicle,other_vehicle_dict,camera_angle,save_path:Path,plot,sim_count): #maybe set other vehicle to a list

    save_path.mkdir(exist_ok=True)
    cam_altitude,other_altitude = altitude_type()
    speed = 25
    frame_data = {}
    center_data = client.getMultirotorState(vehicle_name=camera_vehicle)
    center_cam = (center_data.kinematics_estimated.position.x_val,
                  center_data.kinematics_estimated.position.y_val,
                  cam_altitude)
    center_other = (center_data.kinematics_estimated.position.x_val,
                    center_data.kinematics_estimated.position.y_val,
                    other_altitude)
    radius = random.randint(400,800)
    cam_travel = generate_travel(center_cam,radius,speed)
    other_travel = generate_travel(center_other,radius,speed)
    estimate_trajectory(center_cam,radius,cam_travel,other_travel,plot,save_path,sim_count)
    #plot in 3D with height diff
    # set weather
    weather = random.choice([WeatherParameter.Nothing,WeatherParameter.Nothing,WeatherParameter.Rain,
                             WeatherParameter.Snow])
    weather_val = random.randint(20,80) /100
    change_weather(client,weather,weather_val)
    #print("wheather : ",weather," | intensity :",weather_val)
    #setup vehicle
    client.enableApiControl(True, vehicle_name=camera_vehicle)
    client.armDisarm(True, vehicle_name=camera_vehicle)
    client.enableApiControl(True, vehicle_name=other_vehicle_dict["Name"])
    client.armDisarm(True, vehicle_name=other_vehicle_dict["Name"])
    teleport(client, camera_vehicle, cam_travel.start_coordinates)
    teleport(client,other_vehicle_dict["Name"],other_travel.start_coordinates)
    other_arrived = False
    cam_arrived = False
    pic_count = 1
    pic_delay = 2 #second now but change to fps next
    client.simPause(True)
    save_img(client, pic_count, camera_angle, save_path, sim_count)
    frame_data[str(pic_count)] = get_frame_data(client, camera_vehicle, other_vehicle_dict,camera_angle)
    client.simPause(False)
    while other_arrived == False and cam_arrived == False:
        pic_count += 1
        #print(pic_count)

        cam_arrived = set_next_position(client,cam_travel,camera_vehicle,pic_delay,"osef")

        other_arrived = set_next_position(client, other_travel, other_vehicle_dict["Name"], pic_delay, "osef")
        client.simPause(True)
        save_img(client,pic_count,camera_angle, save_path,sim_count)
        frame_data[str(pic_count)] = get_frame_data(client, camera_vehicle, other_vehicle_dict,camera_angle)
        client.simPause(False)
    #client.armDisarm(False, vehicle_name=other_vehicle)
    #client.armDisarm(False, vehicle_name=camera_vehicle)
    #go_to_start(client,[camera_vehicle]) #not useful
    time.sleep(5)
    return log_dict(cam_travel, other_travel,weather,weather_val,center_cam,center_other,radius,other_vehicle_dict,frame_data)

def create_folders(save_path:Path,vehicle_list:list):

    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)
    for i in vehicle_list:
        create = save_path / i
        if not create.exists():
            create.mkdir()


def dataset_generation(save_path:Path, vehicle_dict:dict, vehicle_used, sim_count_per_vehicle,camera_angle:list):
    # set randomness
    seed_value = random.randint(0,10000)
    random.seed(seed_value)

    # Create save folder if doesn't exist

    create_folders(save_path,vehicle_used)

    vehicle_dict = {k: v for k, v in vehicle_dict.items() if any(value in vehicle_used for value in v.values())}
    # Create client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    # Create vehicles
    create_all_vehicles(client,vehicle_dict)

    # Start simulation
    sim_log = {}
    sim_log["random_seed"] = seed_value
    #get setting
    camera_vehicle = "camera_vehicle"

    # Set detector

    detection_radius_m = 1000
    setdetection(client, camera_vehicle,camera_angle,vehicle_dict,detection_radius_m)
    sim_count = 1 #create count with index in for
    for loop in tqdm(range(1,sim_count_per_vehicle+1)):
        for vehicle_key in tqdm(vehicle_dict.keys()):
            print("Vehicle :", vehicle_dict[vehicle_key]["Name"])
            sim_log[str(sim_count)] = simulation(client,camera_vehicle,vehicle_dict[vehicle_key],
                                         camera_angle,save_path / vehicle_dict[vehicle_key]["Name"] / "{:02d}".format(sim_count),False,sim_count)
            client.reset() #else the balloon goes flying and generate collisions
            go_to_graveyard(client, vehicle_dict)#{1:vehicle_dict[other_vehicle]})
            sim_count += 1
    json_object = json.dumps(sim_log, indent=4)
    with open(str(save_path / "sim_log.json"), "w") as outfile:
        outfile.write(json_object)


