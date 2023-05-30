import math
import time
from pathlib import Path
import shutil
import random
import numpy as np
import airsim
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def save_img(client, counter,camera_angle, save_path):
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
    client.simPause(True)
    img_list = []
    for i in camera_angle:
        response = client.simGetImages([airsim.ImageRequest(i, airsim.ImageType.Scene, False, False)])
        #print("img ", i)
        img_list.append(response[0])
    for i in range(len(camera_angle)):
        #print("save img ", camera_angle[i])
        img1d = np.frombuffer(img_list[i].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(img_list[i].height, img_list[i].width, 3)
        airsim.write_png(str(save_path / (counter + '_' + camera_angle[i] + '.png ')), img_rgb)
        #airsim.write_png(r'data\image\created_data\\' + counter + '_' + camera_angle[i] + '.png ', img_rgb)
    client.simPause(False)


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
    pose = airsim.Pose(airsim.Vector3r(0,0,0))
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
    client.moveToZAsync(coordinates[2], 10,vehicle_name=vehicle_name).join()
    client.takeoffAsync(vehicle_name=vehicle_name).join()
    client.hoverAsync(vehicle_name).join()


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
        if distance >= radius*1.2:
            flag_farenough = True

    altitude_change = random.uniform(-1,1) * center[2] * random.randint(5,20) / 100
    start[2] = start[2] + round(altitude_change/2)
    end[2] = end[2] - round(altitude_change/2)
    trip = travel(start,end,speed,angle_start,angle_end)

    return trip


def estimate_trajectory(center,radius,travel1: travel,travel2: travel,plot:bool,save_path:Path):
    fig, ax = plt.subplots(figsize=(5,5))

    circle = plt.Circle((center[0],center[1]),radius,edgecolor='red',facecolor='none')
    ax.add_patch(circle)
    # Add an arrow to the line
    arrow_head_length = 40

    ax.arrow(travel1.start_coordinates[0], travel1.start_coordinates[1],
             travel1.end_coordinates[0] - travel1.start_coordinates[0],
             travel1.end_coordinates[1] - travel1.start_coordinates[1],
             width=8, head_width=40, head_length=arrow_head_length, fc='blue', ec='blue')
    ax.arrow(travel2.start_coordinates[0], travel2.start_coordinates[1],
             travel2.end_coordinates[0] - travel2.start_coordinates[0],
             travel2.end_coordinates[1] - travel2.start_coordinates[1],
             width=8, head_width=40, head_length=arrow_head_length, fc='green', ec='green')
    plt.savefig(save_path / "estimated_path.png")
    if plot:
        plt.show()

def log_dict(travel_cam,travel_other, weather_param, weather_val,center,radius, vehicle_list):
    log = {}
    log["vehicle_list"] = vehicle_list
    log["circle"] = {"center": center,
                     "radius": radius}
    log["travel_cam"] = travel_cam
    log["travel_other"] = travel_other
    log["weather"] = {"param": weather_param,
                      "val": weather_val}
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


def create_all_vehicles(client,vehicle_list):
    for i in vehicle_list:
        create_vehicle(client, i, i)
    go_to_graveyard(client,vehicle_list)


def go_to_graveyard(client,vehicle_list):
    graveyard_coord = [0,0,60]
    pose = airsim.Pose(airsim.Vector3r(graveyard_coord[0],
                                       graveyard_coord[1],
                                       graveyard_coord[2]))
    for i in vehicle_list:
        client.simSetVehiclePose(pose,True,vehicle_name=i)

def go_to_start(client,vehicle_list):
    graveyard_coord = [0, 0, 0]
    pose = airsim.Pose(airsim.Vector3r(graveyard_coord[0],
                                       graveyard_coord[1],
                                       graveyard_coord[2]))
    for i in vehicle_list:
        client.simSetVehiclePose(pose,True,vehicle_name=i)


def altitude_type():
    diff_list = [-150,0,150]
    diff = random.choice(diff_list)
    altitude = random.randint(-600, -300)
    return altitude, altitude+diff



def simulation(client,camera_vehicle,other_vehicle,camera_angle,save_path:Path,plot = True): #maybe set other vehicle to a list

    cam_altitude,other_altitude = altitude_type()
    speed = 25
    center_data = client.getMultirotorState(vehicle_name=camera_vehicle)
    center_cam = (center_data.kinematics_estimated.position.x_val,
                  center_data.kinematics_estimated.position.x_val,
                  cam_altitude)
    center_other = (center_data.kinematics_estimated.position.x_val,
                    center_data.kinematics_estimated.position.x_val,
                    other_altitude)
    radius = random.randint(200,400)
    cam_travel = generate_travel(center_cam,radius,speed)
    other_travel = generate_travel(center_other,radius,speed)
    estimate_trajectory(center_cam,radius,cam_travel,other_travel,plot,save_path)
    #plot in 3D with height diff
    # set weather
    weather = random.choice([WeatherParameter.Nothing,WeatherParameter.Nothing,WeatherParameter.Rain,
                             WeatherParameter.Snow])
    weather_val = random.randint(20,80) /100
    change_weather(client,weather,weather_val)
    #print("wheather : ",weather," | intensity :",weather_val)
    #setup cam vehicle
    client.enableApiControl(True, vehicle_name=camera_vehicle)
    client.armDisarm(True, vehicle_name=camera_vehicle)
    teleport(client, camera_vehicle, cam_travel.start_coordinates)
    #setup second
    client.enableApiControl(True, vehicle_name=other_vehicle)
    client.armDisarm(True,vehicle_name=other_vehicle)
    teleport(client,other_vehicle,other_travel.start_coordinates)
    other_arrived = False
    cam_arrived = False
    count = 1
    pic_delay = 1 #second now but change to fps next
    while other_arrived == False and cam_arrived == False:
        count += 1
        print(count)

        cam_arrived = set_next_position(client,cam_travel,camera_vehicle,pic_delay,"end")
        other_arrived = set_next_position(client, other_travel, other_vehicle, pic_delay, "random")
        save_img(client,str(count),camera_angle, save_path)
    client.armDisarm(False, vehicle_name=other_vehicle)
    client.armDisarm(False, vehicle_name=camera_vehicle)
    go_to_start(client,[camera_vehicle]) #not useful
    go_to_graveyard(client,[other_vehicle])

def create_folders(save_path:Path,vehicle_list:list):

    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)
    for i in vehicle_list:
        create = save_path / i
        if not create.exists():
            create.mkdir()


def dataset_generation(save_path:Path, vehicle_list, number_of_sim,camera_angle:list):
    # set randomness
    seed_value = random.randint(0,10000)
    random.seed(seed_value)

    # Create save folder if doesn't exist

    create_folders(save_path,vehicle_list)

    # Create client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    # Create vehicles
    create_all_vehicles(client,vehicle_list)

    # Start simulation
    camera_vehicle = "camera_vehicle"
    for i in tqdm(range(number_of_sim)):
        other_vehicle = i % len(vehicle_list)
        print("Vehicle :",other_vehicle)
        simulation(client,camera_vehicle,vehicle_list[other_vehicle],camera_angle,save_path / vehicle_list[other_vehicle])



