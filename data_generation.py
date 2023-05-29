import math
import time
import random
import numpy as np
import airsim
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


def save_img(client, counter,camera_angle = ("front", "left", "right", "back"), save_path = r'data\image\created_data\\'):
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
        print("img ", i)
        img_list.append(response[0])
    for i in range(len(camera_angle)):
        print("save img ", camera_angle[i])
        img1d = np.frombuffer(img_list[i].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(img_list[i].height, img_list[i].width, 3)
        airsim.write_png(save_path + counter + '_' + camera_angle[i] + '.png ', img_rgb)
        #airsim.write_png(r'data\image\created_data\\' + counter + '_' + camera_angle[i] + '.png ', img_rgb)
    client.simPause(False)


def create_vehicle(client, pawn_path):
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
    pose = airsim.Pose(airsim.Vector3r(3,3,0))
    client.simAddVehicle(pawn_path, "simpleflight", pose, pawn_path)
    client.enableApiControl(True, vehicle_name=pawn_path)


def teleport(client, vehicle_name, coordinates):
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
                                       coordinates[2]))
    client.simSetVehiclePose(pose, True, vehicle_name)
    client.takeoffAsync().join()
    client.hoverAsync(vehicle_name).join()


def get_coordinates(center,radius, angle):


    x = center[0] + radius * math.cos(math.radians(angle))
    y = center[1] + radius * math.sin(math.radians(angle))
    coordinates = [x,y,center[2]]
    return coordinates


def get_distance(coord1,coord2):
    return math.sqrt((coord2[0]-coord1[0])**2 + (coord2[1]-coord1[1])**2)


def generate_travel(center,radius):
    flag_farenough = False
    angle_start = random.randint(1,360)
    start = get_coordinates(center,radius,angle_start)
    while not flag_farenough:
        angle_end = random.randint(0,360)
        end = get_coordinates(center,radius,angle_end)
        distance = get_distance(start,end)
        print(distance)
        if distance >= radius*1.2:
            flag_farenough = True

    altitude_change = random.uniform(-1,1) * center[2] * random.randint(5,20) / 100
    start[2] = start[2] + round(altitude_change/2)
    end[2] = end[2] - round(altitude_change/2)
    trip = travel(start,end,25,angle_start,angle_end)

    return trip


def estimate_trajectory(center,radius,travel1: travel,travel2: travel):
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

def simulation(camera_vehicle,other_vehicle): #maybe set other vehicle to a list
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    #set_seed
    seed_value = random.randint(0,10000)
    random.seed(seed_value)
    #player start place in map center
    #teleport before spawning other_vehicle
    altitude = random.randint(-600,-300)
    print(altitude)
    center_data = client.getMultirotorState(vehicle_name=camera_vehicle)

    center = (center_data.kinematics_estimated.position.x_val,
              center_data.kinematics_estimated.position.x_val,
              altitude)
    radius = random.randint(200,400)
    cam_travel = generate_travel(center,radius)
    other_travel = generate_travel(center,radius)
    estimate_trajectory(center,radius,cam_travel,other_travel)

    # set weather
    weather = random.choice([WeatherParameter.Nothing,WeatherParameter.Nothing,WeatherParameter.Rain,
                             WeatherParameter.Snow])
    weather_val = random.randint(20,80) /100
    change_weather(client,weather,weather_val)
    print("wheather : ",weather," | intensity :",weather_val)
    create_vehicle(client,other_vehicle)
    client.enableApiControl(True, vehicle_name=camera_vehicle)
    client.enableApiControl(True, vehicle_name=other_vehicle)
    client.armDisarm(True,vehicle_name=camera_vehicle)
    client.armDisarm(True,vehicle_name=other_vehicle)

    client.takeoffAsync(vehicle_name=camera_vehicle).join()
    client.takeoffAsync(vehicle_name=other_vehicle).join()

    teleport(client,camera_vehicle,cam_travel.start_coordinates)
    teleport(client,other_vehicle,other_travel.start_coordinates)
    cam = client.moveToPositionAsync(cam_travel.end_coordinates[0],
                                     cam_travel.end_coordinates[1],
                                     cam_travel.end_coordinates[2],
                                     cam_travel.speed,
                                     vehicle_name=camera_vehicle,
                                     drivetrain=airsim.DrivetrainType.ForwardOnly,
                                     yaw_mode=airsim.YawMode(False,0))

    other = client.moveToPositionAsync(other_travel.end_coordinates[0],
                               other_travel.end_coordinates[1],
                               other_travel.end_coordinates[2],
                               other_travel.speed,
                               vehicle_name=other_vehicle,
                               drivetrain=airsim.DrivetrainType.ForwardOnly,
                               yaw_mode=airsim.YawMode(False, 0))

    count = 1
    cam_data = client.getMultirotorState(vehicle_name=camera_vehicle)
    cam_coord = [cam_data.kinematics_estimated.position.x_val,cam_data.kinematics_estimated.position.y_val]

    other_data = client.getMultirotorState(vehicle_name=other_vehicle)
    other_coord = [other_data.kinematics_estimated.position.x_val, other_data.kinematics_estimated.position.y_val]

    while get_distance(other_coord,other_travel.end_coordinates) > 30 or get_distance(cam_coord,cam_travel.end_coordinates) > 30:

        print(count)
        # save_img(client,str(count))
        count += 1
        time.sleep(1)
        cam_data = client.getMultirotorState(vehicle_name=camera_vehicle)
        cam_coord = [cam_data.kinematics_estimated.position.x_val, cam_data.kinematics_estimated.position.y_val]
        other_data = client.getMultirotorState(vehicle_name=other_vehicle)
        other_coord = [other_data.kinematics_estimated.position.x_val, other_data.kinematics_estimated.position.y_val]
        print(get_distance(other_coord,other_travel.end_coordinates))
        print(get_distance(cam_coord,cam_travel.end_coordinates))
    cam.join()
    other.join()
    client.reset()





