from __future__ import division

import math

import numpy as np
import cv2
import os
import sys
import time
import airsim
import racing_utils
import vel_regressor
import airsim.utils
from airsimdroneracingvae.types import Pose

# import utils
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)

# policy options: bc_con, bc_unc, bc_img, bc_reg, bc_full
policy_type = 'bc_img'
gate_noise = 1
lateral_noise = 10.0


def process_image(client, img_res):
    image_response = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
    img_1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_bgr = img_1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
    img_resized = cv2.resize(img_bgr, (img_res, img_res)).astype(np.float32)
    img_batch_1 = np.array([img_resized])
    cam_pos = image_response.camera_position
    cam_orientation = image_response.camera_orientation
    return img_batch_1, cam_pos, cam_orientation


def move_drone(client, vel_cmd):
    vel_cmd = np.array(vel_cmd)
    # good multipliers originally: 0.4 for vel, 0.8 for yaw
    # good multipliers new policies: 0.8 for vel, 0.8 for yaw
    vel_cmd[0:2] = vel_cmd[0:2] * 1.5  # usually base speed is 3/ms
    vel_cmd[3] = vel_cmd[3] * 1.5
    # yaw rate is given in deg/s!! not rad/s
    yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=vel_cmd[3] * 180.0 / np.pi)
    client.moveByVelocityAsync(vel_cmd[0], vel_cmd[1], vel_cmd[2], duration=0.1, yaw_mode=yaw_mode)


print(os.path.abspath(airsim.__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

if __name__ == "__main__":
    client = airsim.MultirotorClient()
    client.confirmConnection()
    time.sleep(2)
    drone_name = "drone_0"
    client.enableApiControl(True, vehicle_name=drone_name)
    client.armDisarm(True, vehicle_name=drone_name)

    racing_utils.trajectory_utils.AllGatesDestroyer(client)
    offset = [0, 0, -0]

    # gate_poses = racing_utils.trajectory_utils.RedGateSpawnerCircle(client, num_gates=8, radius=10, radius_noise=gate_noise, height_range=[0, -gate_noise], track_offset=offset)
    # gate_poses = racing_utils.trajectory_utils.LineGateSpawner(client, num_gates=8, spacing=7, height_noise=gate_noise, lateral_noise=lateral_noise)
    gate_poses = racing_utils.trajectory_utils.SGateSpawner(client, num_gates=8, spacing=8, height_noise=gate_noise, lateral_noise=lateral_noise)

    # 设置起飞位置和姿态
    # position of circle curve
    # takeoff_position = airsim.Vector3r(-1.2, -7.5, -1.0)  # 通过这个来调整初始位置比较好
    # takeoff_orientation = airsim.Quaternionr(0.4, 0.9, 0)

    # takeoff_position = airsim.Vector3r(-2, -13.5, -1.4)  # 通过这个来调整初始位置比较好
    # takeoff_orientation = airsim.Quaternionr(0.4, 0.9, 0)

    # # position of line gate
    takeoff_position = airsim.Vector3r(-1.2, 2.0, -1.2)
    takeoff_orientation = airsim.Quaternionr(0, 0, 0)

    # 起飞
    client.takeoffAsync(vehicle_name=drone_name).join()

    # 移动到指定位置和姿态
    client.moveToPositionAsync(takeoff_position.x_val, takeoff_position.y_val, takeoff_position.z_val, 5, vehicle_name=drone_name).join()
    client.moveByVelocityAsync(0, 0, 0, duration=1, vehicle_name=drone_name).join()  # 停止移动
    # 设置姿态
    yaw = airsim.to_eularian_angles(takeoff_orientation)[2]
    client.rotateByYawRateAsync(yaw, 5, vehicle_name=drone_name).join()

    time.sleep(1.0)
    img_res = 64

    if policy_type == 'bc_con':
        training_mode = 'latent'
        latent_space_constraints = True
        bc_weights_path = 'E:/AirSim-Drone-Racing-VAE-Imitation/model_outputs/bc_con/bc_model_150.ckpt'
        feature_weights_path = 'E:/AirSim-Drone-Racing-VAE-Imitation/model_outputs/cmvae_con/cmvae_model_40.ckpt'
    elif policy_type == 'bc_unc':
        training_mode = 'latent'
        latent_space_constraints = False
        bc_weights_path = 'E:/AirSim-Drone-Racing-VAE-Imitation/model_outputs/bc_unc/bc_model_150.ckpt'
        feature_weights_path = 'E:/AirSim-Drone-Racing-VAE-Imitation/model_outputs/cmvae_unc/cmvae_model_45.ckpt'
    elif policy_type == 'bc_img':
        training_mode = 'latent'
        latent_space_constraints = True
        bc_weights_path = 'E:/AirSim-Drone-Racing-VAE-Imitation/model_outputs/bc_img/bc_model_100.ckpt'
        feature_weights_path = 'E:/AirSim-Drone-Racing-VAE-Imitation/model_outputs/cmvae_img/cmvae_model_45.ckpt'
    elif policy_type == 'bc_reg':
        training_mode = 'reg'
        latent_space_constraints = True
        bc_weights_path = 'E:/AirSim-Drone-Racing-VAE-Imitation/model_outputs/bc_reg/bc_model_80.ckpt'
        feature_weights_path = 'E:/AirSim-Drone-Racing-VAE-Imitation/model_outputs/reg/reg_model_25.ckpt'
    elif policy_type == 'bc_full':
        training_mode = 'full'
        latent_space_constraints = True
        bc_weights_path = 'E:/AirSim-Drone-Racing-VAE-Imitation/model_outputs/bc_full/bc_model_120.ckpt'
        feature_weights_path = None

    vel_regressor = vel_regressor.VelRegressor(regressor_type=training_mode, bc_weights_path=bc_weights_path,
                                               feature_weights_path=feature_weights_path,
                                               latent_space_constraints=latent_space_constraints)
    count = 0
    max_count = 50
    times_net = np.zeros((max_count,))
    times_loop = np.zeros((max_count,))
    while True:
        start_time = time.time()
        img_batch_1, cam_pos, cam_orientation = process_image(client, img_res)
        elapsed_time_net = time.time() - start_time
        times_net[count] = elapsed_time_net
        p_o_b = Pose(cam_pos, cam_orientation)
        vel_cmd = vel_regressor.predict_velocities(img_batch_1, p_o_b)
        move_drone(client, vel_cmd)
        elapsed_time_loop = time.time() - start_time
        times_loop[count] = elapsed_time_loop
        count = count + 1
        if count == max_count:
            count = 0
            avg_time = np.mean(times_net)
            avg_freq = 1.0 / avg_time
            print('Avg network time over {} iterations: {} ms | {} Hz'.format(max_count, avg_time * 1000, avg_freq))
            avg_time = np.mean(times_loop)
            avg_freq = 1.0 / avg_time
            print('Avg loop time over {} iterations: {} ms | {} Hz'.format(max_count, avg_time * 1000, avg_freq))
