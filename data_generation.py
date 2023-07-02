import math
import time
from pathlib import Path
import shutil
import cProfile
from skimage import color
import random
import numpy as np
import airsim
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import cv2
import winsound

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


def save_img(client, frame,camera_angle, save_path,sim_count,camera_vehicle,vehicle_dict):
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
        response = client.simGetImages([airsim.ImageRequest(i, airsim.ImageType.Scene, False, False)],"",False)
        #print("img ", i)
        img_list.append(response[0])
    label = checkbb_camera(client, camera_vehicle, camera_angle, vehicle_dict)
    if label:
        for i in range(len(camera_angle)):
            #print("save img ", camera_angle[i])
            img1d = np.frombuffer(img_list[i].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(img_list[i].height, img_list[i].width, 3)
            gray_image = color.rgb2gray(img_rgb) * 255
            frame_path = str(save_path / ("{:02d}".format(sim_count) + '_' + "{:03d}".format(frame) + '_' + camera_angle[i] + '.png '))
            cv2.imwrite(frame_path, gray_image)
    return label



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
    #client.takeoffAsync(vehicle_name=vehicle_name).join()
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


def estimate_trajectory(center,radius,travel1: travel,listtravel2: list[travel],plot:bool,save_path:Path,sim_count):
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
    for travel2 in listtravel2:
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.quiver(
            travel2.start_coordinates[0], travel2.start_coordinates[1], travel2.start_coordinates[2],
            travel2.end_coordinates[0] - travel2.start_coordinates[0],
            travel2.end_coordinates[1] - travel2.start_coordinates[1],
            travel2.end_coordinates[2] - travel2.start_coordinates[2],
            arrow_length_ratio=0.1, color='blue')
    plt.savefig(save_path / ("{:02d}".format(sim_count) +"_estimated_path.png"))
    if plot:
        plt.show()
    plt.close()

def log_dict(cam_travel,list_travel, weather_param, weather_val, vehicle_dict,frame_data):
    log = {}
    log["travel_cam"] = {"start": cam_travel.start_coordinates,
                         "end": cam_travel.end_coordinates,
                         "speed":cam_travel.speed,
                         "angle_start": cam_travel.angle_start,
                         "angle_end": cam_travel.angle_end
                         }
    for travel_id in range(0, len(list_travel)):
        list_key = list(vehicle_dict.keys())
        log["travel"+vehicle_dict[list_key[travel_id]]["Name"]] = {"start": list_travel[travel_id].start_coordinates,
                             "end": list_travel[travel_id].end_coordinates,
                             "speed":list_travel[travel_id].speed,
                             "angle_start": list_travel[travel_id].angle_start,
                             "angle_end": list_travel[travel_id].angle_end
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
        next_x = x + (xf - x) * (travel.speed * t) / distance_left
        next_y = y + (yf - y) * (travel.speed * t) / distance_left
        next_z = z + (zf - z) * (travel.speed * t) / distance_left
        teleport(client, vehicle_name,
                 [next_x, next_y, next_z],
                 [pitch, roll, yaw])
        return False


def create_all_vehicles(client,vehicle_dict):
    for key in vehicle_dict.keys():
        create_vehicle(client, vehicle_dict[key]["PawnPath"], vehicle_dict[key]["Name"])
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


def altitude_type(altitude):
    diff_list = [-100,-50,0,50,100]
    diff = random.choice(diff_list)
    return altitude+diff

def setdetection(client, camera_vehicle, camera_angle,vehicle_dict,detection_radius_m):
    radius_detect_cm = detection_radius_m * 100
    for cam_name in camera_angle:
        client.simSetDetectionFilterRadius(cam_name, airsim.ImageType.Scene, radius_detect_cm, camera_vehicle)
        for key in vehicle_dict.keys():
            client.simAddDetectionFilterMeshName(cam_name, airsim.ImageType.Scene, vehicle_dict[key]["Mesh"] + "*", camera_vehicle)

def find_key_by_name(dictionary, name):
    for key, value in dictionary.items():
        if value.get("Name") == name:
            return key

def check_overlap(bounding_boxes):
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

def checkbb_camera(client,camera_vehicle,camera_angle,vehicle_dict): #manage if vehicle name different than vehicle mesh
    label = {}
    for i in camera_angle:
        info_list = client.simGetDetections(i,airsim.ImageType.Scene,camera_vehicle)
        if not info_list:
            continue
        for info in info_list:
            distance = get_distance([0,0,0],[info.relative_pose.position.x_val,
                                             info.relative_pose.position.y_val,
                                             info.relative_pose.position.z_val])
            vehicle_key = find_key_by_name(vehicle_dict,info.name)
            if distance >= vehicle_dict[vehicle_key]["Max_dist_m"]:
                continue
            label[i + "_" + info.name] = {"name": info.name,
                        "yolo_class": vehicle_dict[vehicle_key]["yolo_class"],
                        "box2D": {"x_max": info.box2D.max.x_val,
                                  "y_max": info.box2D.max.y_val,
                                  "x_min": info.box2D.min.x_val,
                                  "y_min": info.box2D.min.y_val}}
        if len(info_list) >1:
            on_same_picture_list = {key: value for key, value in label.items() if key.startswith(i)}
            bbx_on_same_picture = []
            for key in on_same_picture_list.keys():
                bbx_on_same_picture.append(on_same_picture_list[key]["box2D"])
            if check_overlap(bbx_on_same_picture):
                for key in on_same_picture_list.keys():
                    del label[key]
    return label


def get_frame_data(client,camera_vehicle, bbx_dict,timeoftheday,current_level):


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
    result = {"currentlevel":current_level, "timeoftheday":timeoftheday,camera_vehicle:v1_log}
    name_list = [(details[1]["name"],details[0]) for details in bbx_dict.items()]
    for name in name_list:
        v2_data = client.getMultirotorState(vehicle_name=name[0])
        v2_yaw = v2_data.kinematics_estimated.orientation.z_val
        v2_log = {"coordinate":
                  [v2_data.kinematics_estimated.position.x_val,
                   v2_data.kinematics_estimated.position.y_val,
                   v2_data.kinematics_estimated.position.z_val],
              "orientation":
                   [v2_data.kinematics_estimated.orientation.x_val,
                    v2_data.kinematics_estimated.orientation.y_val,
                    v2_data.kinematics_estimated.orientation.z_val]}

        v2_log["distance"]= get_distance(v1_log["coordinate"],v2_log["coordinate"]),
        v2_log["angle"] = np.degrees(v2_yaw - v1_yaw) #positive diff angle = clockwise
        v2_log["label"] = bbx_dict[name[1]]
        result[name[1]] = v2_log

    return result

def add_to_count(count_dict,bbx_dict):

    for key,value in bbx_dict.items():
        class_found = value["yolo_class"]
        count_dict[class_found] += 1

def simulation(client,camera_vehicle,vehicle_dict,camera_angle,save_path:Path,plot,sim_count,current_level,timeoftheday_dict, count_dict): #maybe set other vehicle to a list

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
    cam_travel = generate_travel(center_cam,cam_radius,speed)
    for vehicle in vehicle_dict.keys():
        other_altitude = altitude_type(cam_altitude)
        center_other=(center_data.kinematics_estimated.position.x_val,
                      center_data.kinematics_estimated.position.y_val,
                      other_altitude)
        max_dist = vehicle_dict[vehicle]["Max_dist_m"]
        radius = random.randint(round(max_dist * 0.9), round(max_dist * 1.1))
        list_travel.append(generate_travel(center_other,radius,speed))

    estimate_trajectory(center_cam,cam_radius,cam_travel,list_travel,plot,save_path,sim_count)
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
    for vehicle in vehicle_dict.keys():
        client.enableApiControl(True, vehicle_name=vehicle_dict[vehicle]["Name"])
        client.armDisarm(True, vehicle_name=vehicle_dict[vehicle]["Name"])

    other_arrived = False
    cam_arrived = False
    pic_count = 1
    pic_delay = 2 #second now but change to fps next
    #set timeoftheday
    rand_time = random.choice(list(timeoftheday_dict.keys()))
    client.simSetTimeOfDay(True, timeoftheday_dict[rand_time], False, 1, 1000)  # sun not moving for a whole
    teleport(client, camera_vehicle, cam_travel.start_coordinates)
    travel_counter = 0
    for vehicle in vehicle_dict.keys():
        teleport(client, vehicle_dict[vehicle]["Name"], list_travel[travel_counter].start_coordinates)
        travel_counter +=1
    time.sleep(1)
    while other_arrived == False and cam_arrived == False:
        travel_counter = 0
        #print(pic_count)
        # set time of the day
        rand_time = random.choice(list(timeoftheday_dict.keys()))
        client.simSetTimeOfDay(True, timeoftheday_dict[rand_time], False, 1, 1000)  # sun not moving for a whole iteration
        cam_arrived = set_next_position(client,cam_travel,camera_vehicle,pic_delay,None)
        for vehicle in vehicle_dict.keys():
            flag_travel = set_next_position(client, list_travel[travel_counter], vehicle_dict[vehicle]["Name"], pic_delay, "random")
            if flag_travel == True:
                other_arrived = True
            travel_counter +=1
        time.sleep(0.5)
        client.simPause(True)
        bbx_dict = save_img(client, pic_count, camera_angle, save_path, sim_count, camera_vehicle, vehicle_dict)
        add_to_count(count_dict,bbx_dict)
        if bbx_dict:
            frame_data[str(pic_count)] = get_frame_data(client, camera_vehicle, bbx_dict,rand_time,current_level)
            pic_count += 1
        client.simPause(False)
        if pic_count >=60:
            cam_arrived = True
    time.sleep(2)
    return log_dict(cam_travel, list_travel, weather, weather_val, vehicle_dict, frame_data)


def create_folders(save_path:Path,vehicle_list:list):

    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)
    for i in vehicle_list:
        create = save_path / i
        if not create.exists():
            create.mkdir()


def initialize_level(client,vehicle_dict,camera_angle,camera_vehicle):
    # Create vehicles
    create_all_vehicles(client,vehicle_dict)
    detection_radius_m = 1000
    setdetection(client, camera_vehicle,camera_angle,vehicle_dict,detection_radius_m)


def change_level(current_sim,sim_num,level_list):
    if sim_num < len(level_list):
        raise ValueError("Number of simulation smaller than list of level")
    num_level = len(level_list)
    list_change = []
    list_change.append(1)
    for i in range(1,num_level):
        list_change.append(list_change[i-1] + (sim_num // num_level))

    if current_sim in list_change:
        return True
    else:
        return False


def dataset_generation(save_path:Path, vehicle_dict:dict, vehicle_used, sim_count_per_vehicle,camera_angle:list,level_list,timeoftheday_dict):
    # set randomness
    seed_value = random.randint(0,10000)
    random.seed(seed_value)

    # Create save folder if doesn't exist
    if not (save_path / "data").exists():
        (save_path / "data").mkdir()
    #create_folders(save_path / "data",vehicle_used)

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
    #get setting
    camera_vehicle = "camera_vehicle"
    dict_camera = {}
    for angle in camera_angle:
        test_img = client.simGetImages([airsim.ImageRequest(angle, airsim.ImageType.Scene, False, False)], "camera_vehicle",False)[0]
        dict_camera[angle] = {"resolution":[test_img.width,test_img.height]}
    sim_log["camera_details"] = dict_camera
    sim_count = 1 #create count with index in for
    #Level Management
    level_idx =0
    current_level = "StartLevel"
    for loop in tqdm(range(1,sim_count_per_vehicle+1)):
        if change_level(loop,sim_count_per_vehicle,level_list):
            current_level = level_list[level_idx]
            time.sleep(2)  # time for engine to load everything
            winsound.PlaySound("mixkit-elevator-tone-2863.wav.wav", winsound.SND_FILENAME)
            input("Change level to "+current_level)
            #client.simLoadLevel(current_level) #Loading Level API is fairly random change manually if needed
            #time.sleep(20) #time for level to load
            # Create client
            client = airsim.MultirotorClient()
            client.confirmConnection()
            client.enableApiControl(True)
            initialize_level(client,vehicle_dict,camera_angle,camera_vehicle)
            level_idx += 1



        sim_log[str(sim_count)] = simulation(client,camera_vehicle,vehicle_dict,
                                             camera_angle,save_path / "data" / "{:02d}".format(sim_count),False,
                                             sim_count,current_level,timeoftheday_dict, count_dict)
        client.reset() #else the balloon goes flying and generate collisions
        go_to_graveyard(client, vehicle_dict)#{1:vehicle_dict[other_vehicle]})
        sim_count += 1
    client.simPause(True)
    sim_log["classes_count"] = count_dict
    json_object = json.dumps(sim_log, indent=4)
    with open(str(save_path / "sim_log.json"), "w") as outfile:
        outfile.write(json_object)
    order_dataset(save_path,vehicle_dict)
    generate_train_val_test(save_path / "ordered_data",(80,10,10))

def copy_file(source_path: Path,destination_directory: Path,):
    filename = source_path.name
    shutil.copy(source_path, destination_directory / filename)
    #time.sleep(0.3) #due to hard drive limitations this sleep makes the copy more fluid
def convert_to_yolo_label(xmax,ymax,xmin,ymin,width,height):
    center_x = (xmin + (xmax - xmin)/2) / width
    center_y = (ymin + (ymax - ymin)/2) / height
    height_bbx = (ymax - ymin) / height
    width_bbx = (xmax - xmin) / width

    return center_x, center_y, width_bbx, height_bbx

def check_full(xmin,xmax,ymin,ymax,width,height):
    if xmin == 0 or xmax == width or ymin== 0 or ymax == height:
        return False
    else:
        return True

def order_dataset(save_path,vehicle_dict):
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


    count_background=0
    # Order
    for sim_number in tqdm(sim_log.keys()):
        for frame_number, frame_data in tqdm(sim_log[sim_number]['frame_data'].items()):

            list_img_frame = list((save_path / "data" / "{:02d}".format(sim_number)).glob("{:02d}".format(sim_number) + '_' + "{:03d}".format(int(frame_number)) + '_*'))
            list_vehicle_key = list(frame_data.keys())[3:]
            for key in list_vehicle_key:
                vehicle_frame_data = frame_data[key]
                for frame in list_img_frame:
                    angle = key.split("_")[0]
                    if angle in str(frame):
                        xmin = vehicle_frame_data['label']["box2D"]["x_min"]
                        xmax =vehicle_frame_data['label']["box2D"]["x_max"]
                        ymin = vehicle_frame_data['label']["box2D"]["y_min"]
                        ymax = vehicle_frame_data['label']["box2D"]["y_max"]
                        width = resolution_dict[angle]["resolution"][0]
                        height = resolution_dict[angle]["resolution"][1]
                        if check_full(xmin,xmax,ymin,ymax,width,height): #remove object not fully on the picture
                            copy_file(frame,new_ordered_path / "img")
                            x_center, y_center, width, height = convert_to_yolo_label(xmax,ymax,xmin,ymin,width,height)
                            with open(new_ordered_path / "img" / (frame.stem + '.txt'), 'a') as file:
                            # remove label if photo is cropped ?
                                file.write(vehicle_frame_data['label']['yolo_class'] +" "+
                                       str(x_center) +" "+ str(y_center)+" "+
                                       str(width)+ " "+str(height) + "\n")
                    else:
                        count_background +=1
                        if count_background ==  3: #avoid copying all the files
                            count_background = 0
                            copy_file(frame, new_ordered_path / "background")




def create_data_split_folder(folder_name, data_path,dict_class,index_interval):
    save_path = data_path.parent / folder_name
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)
    (save_path / "images").mkdir()
    (save_path / "labels").mkdir()
    percentage_background = 5
    divided_background = round((100 / len(list(dict_class.keys()))) / percentage_background)
    reduced_background_counter = divided_background -1
    for key in tqdm(dict_class.keys()):
        for file in range(index_interval[0],index_interval[1]+1):
            if key == "background": #no label for background
                reduced_background_counter +=1
                if reduced_background_counter == divided_background: #reduce the number of background compare to the rest
                    reduced_background_counter=0
                    copy_file(dict_class[key][file].with_suffix(".png"), save_path / "images")
            else:
                copy_file(dict_class[key][file].with_suffix(".txt"),save_path / "labels")
                copy_file(dict_class[key][file].with_suffix(".png"), save_path / "images")


def generate_train_val_test(data_path: Path,split: tuple):

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

    #create list for each set
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

    for key,value in dict_class.items():
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







