# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import os
import math
import random

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R


class RealManInspireBlockAssemblySearch:

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.device_type = device_type
        self.device_id = device_id
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
        self.headless = headless
        self.graphics_device_id = self.device_id

        self.dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.fingertip_names = ["R_index_distal", "R_middle_distal", "R_ring_distal", "R_pinky_distal", "R_thumb_distal"]
        self.fingertip_adjustment_params = [[[0.15, 0.8, 0.15], 0.05], [[0.15, 0.8, 0.15], 0.055], [[0.2, 0.8, 0.15], 0.05], [[0.2, 0.8, 0.15], 0.045], [[0, 1, 0], 0.02]]

        self.cfg["env"]["numObservations"] = 129
        self.cfg["env"]["numStates"] = 129
        self.cfg["env"]["numActions"] = 13

        self.gym = gymapi.acquire_gym()

        self.num_envs = self.cfg["env"]["numEnvs"]
        self.num_obs = self.cfg["env"]["numObservations"]
        self.num_states = self.cfg["env"]["numStates"]
        self.num_actions = self.cfg["env"]["numActions"]

        self.control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            cam_pos = gymapi.Vec3(0.5, -0.1, 1.5)
            cam_target = gymapi.Vec3(-0.7, -0.1, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.arm_hand_default_dof_pos = to_torch([3.14, 0.6, 0, 0.6, 0., 0.59, -1.571,
             0.0, -0.174, 0.785, 0.0, -0.174, 0.785, 0.0, -0.174, 0.785, 0.0, -0.174, 0.785], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.hand_indices[0], "Link7", gymapi.DOMAIN_ENV)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)
        print("Contact Tensor Dimension", self.contact_tensor.shape)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.segmentation_target_init_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3].clone()
        self.segmentation_target_init_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7].clone()
        self.segmentation_target_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3].clone()
        self.segmentation_target_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7].clone()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim_params.physx.max_gpu_contact_pairs = int(self.sim_params.physx.max_gpu_contact_pairs)

        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        self._create_ground_plane()
        self._create_envs()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = self.cfg["env"]['envSpacing']
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        arm_hand_asset_file = "mjcf/realman_mjcf/realman_inspire_mjmodel.xml"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            arm_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", arm_hand_asset_file)

        # load arm and hand.
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE  # ???
        arm_hand_asset = self.gym.load_asset(self.sim, asset_root, arm_hand_asset_file, asset_options)

        self.num_arm_hand_bodies = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.num_arm_hand_shapes = self.gym.get_asset_rigid_shape_count(arm_hand_asset)
        self.num_arm_hand_dofs = self.gym.get_asset_dof_count(arm_hand_asset)
        self.num_arm_hand_actuators = self.gym.get_asset_actuator_count(arm_hand_asset)
        self.num_arm_hand_tendons = self.gym.get_asset_tendon_count(arm_hand_asset)
        print("self.num_arm_hand_bodies: ", self.num_arm_hand_bodies)
        print("self.num_arm_hand_shapes: ", self.num_arm_hand_shapes)
        print("self.num_arm_hand_dofs: ", self.num_arm_hand_dofs)
        print("self.num_arm_hand_actuators: ", self.num_arm_hand_actuators)
        print("self.num_arm_hand_tendons: ", self.num_arm_hand_tendons)

        # Set up each DOF.
        actuated_dof_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "R_index_MCP_joint", "R_middle_MCP_joint", "R_ring_MCP_joint", "R_pinky_MCP_joint", "R_thumb_MCP_joint2", "R_thumb_MCP_joint1"]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(arm_hand_asset, name) for name in actuated_dof_names]
        actuated_hand_dof_names = ["R_index_MCP_joint", "R_middle_MCP_joint", "R_ring_MCP_joint", "R_pinky_MCP_joint", "R_thumb_MCP_joint2", "R_thumb_MCP_joint1"]
        self.actuated_hand_dof_indices = [self.gym.find_asset_dof_index(arm_hand_asset, name) for name in actuated_hand_dof_names]
        actuated_arm_dof_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.actuated_arm_dof_indices = [self.gym.find_asset_dof_index(arm_hand_asset, name) for name in actuated_arm_dof_names]

        arm_hand_dof_props = self.gym.get_asset_dof_properties(arm_hand_asset)
        arm_hand_dof_props["driveMode"][:] = gymapi.DOF_MODE_POS

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.arm_hand_dof_default_pos = []
        self.arm_hand_dof_default_vel = []

        arm_hand_dof_lower_limits_list = [-3.1, -2.268, -3.1, -2.355, -3.1, -2.233, -6.28]
        arm_hand_dof_upper_limits_list = [3.1, 2.268, 3.1, 2.355, 3.1, 2.233, 6.28]
        arm_hand_dof_default_pos_list = [0.0, 0.0, 0.0, 0.6, 0.0, 0.59, -1.57]
        
        arm_hand_dof_stiffness_list = [200, 200, 100, 100, 50, 50, 50]
        arm_hand_dof_damping_list = [20, 20, 10, 10, 10, 5, 5]
        arm_hand_dof_effort_list = [60, 60, 30, 30, 10, 10, 10]
        arm_hand_dof_velocity_list = [1, 1, 1, 1, 1, 1, 1]

        for i in range(self.num_arm_hand_dofs):
            if i < 7:
                self.arm_hand_dof_lower_limits.append(arm_hand_dof_lower_limits_list[i])
                self.arm_hand_dof_upper_limits.append(arm_hand_dof_upper_limits_list[i])
                self.arm_hand_dof_default_pos.append(arm_hand_dof_default_pos_list[i])
            else:
                self.arm_hand_dof_lower_limits.append(arm_hand_dof_props['lower'][i])
                self.arm_hand_dof_upper_limits.append(arm_hand_dof_props['upper'][i])
                self.arm_hand_dof_default_pos.append(0.0)
            self.arm_hand_dof_default_vel.append(0.0)

            if i < 7:
                arm_hand_dof_props['stiffness'][i] = arm_hand_dof_stiffness_list[i]
                arm_hand_dof_props['effort'][i] = arm_hand_dof_effort_list[i]
                arm_hand_dof_props['damping'][i] = arm_hand_dof_damping_list[i]
                arm_hand_dof_props['velocity'][i] = arm_hand_dof_velocity_list[i]

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.actuated_hand_dof_indices = to_torch(self.actuated_hand_dof_indices, dtype=torch.long, device=self.device)
        self.actuated_arm_dof_indices = to_torch(self.actuated_arm_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(self.arm_hand_dof_lower_limits, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(self.arm_hand_dof_upper_limits, device=self.device)
        self.arm_hand_dof_default_pos = to_torch(self.arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)

        # Put objects in the scene.
        arm_hand_start_pose = gymapi.Transform()
        arm_hand_start_pose.p = gymapi.Vec3(-0.25, 0.0, 0.6)
        arm_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0.0)

        # create table asset
        table_dims = gymapi.Vec3(1.5, 1.0, 0.6)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # create box asset
        box_assets = []
        box_start_poses = []

        box_thin = 0.01
        box_xyz = [0.60, 0.416, 0.165]
        box_offset = [0.25, 0.19, 0]

        box_asset_options = gymapi.AssetOptions()
        box_asset_options.disable_gravity = False
        box_asset_options.fix_base_link = True
        box_asset_options.flip_visual_attachments = True
        box_asset_options.collapse_fixed_joints = True
        box_asset_options.disable_gravity = True
        box_asset_options.thickness = 0.001

        box_bottom_asset = self.gym.create_box(self.sim, box_xyz[0], box_xyz[1], box_thin, table_asset_options)
        box_left_asset = self.gym.create_box(self.sim, box_xyz[0], box_thin, box_xyz[2], table_asset_options)
        box_right_asset = self.gym.create_box(self.sim, box_xyz[0], box_thin, box_xyz[2], table_asset_options)
        box_former_asset = self.gym.create_box(self.sim, box_thin, box_xyz[1], box_xyz[2], table_asset_options)
        box_after_asset = self.gym.create_box(self.sim, box_thin, box_xyz[1], box_xyz[2], table_asset_options)

        box_bottom_start_pose = gymapi.Transform()
        box_bottom_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_thin) / 2)
        box_left_start_pose = gymapi.Transform()
        box_left_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], (box_xyz[1] - box_thin) / 2 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_right_start_pose = gymapi.Transform()
        box_right_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], -(box_xyz[1] - box_thin) / 2 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_former_start_pose = gymapi.Transform()
        box_former_start_pose.p = gymapi.Vec3((box_xyz[0] - box_thin) / 2 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_after_start_pose = gymapi.Transform()
        box_after_start_pose.p = gymapi.Vec3(-(box_xyz[0] - box_thin) / 2 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_xyz[2]) / 2)

        box_assets.append(box_bottom_asset)
        box_assets.append(box_left_asset)
        box_assets.append(box_right_asset)
        box_assets.append(box_former_asset)
        box_assets.append(box_after_asset)
        box_start_poses.append(box_bottom_start_pose)
        box_start_poses.append(box_left_start_pose)
        box_start_poses.append(box_right_start_pose)
        box_start_poses.append(box_former_start_pose)
        box_start_poses.append(box_after_start_pose)

        lego_path = "urdf/blender/urdf/"
        all_lego_files_name = os.listdir("../assets/" + lego_path)

        all_lego_files_name = ['1x2.urdf', '1x2_curve.urdf', '1x3_curve_soft.urdf', '1x3_curve.urdf', '1x1.urdf', '1x3.urdf', '1x4.urdf', '2x2_curve_soft.urdf']
        lego_g_half_length = [0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.03]

        lego_assets = []
        lego_start_poses = []

        for n in range(9):
            for i, lego_file_name in enumerate(all_lego_files_name):
                lego_asset_options = gymapi.AssetOptions()
                lego_asset_options.disable_gravity = False
                lego_asset_options.thickness = 0.00001
                lego_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)

                lego_start_pose = gymapi.Transform()
                if n % 2 == 0:
                    lego_start_pose.p = gymapi.Vec3(-0.17 + 0.17 * int(i % 3) + 0.25, -0.11 + 0.11 * int(i / 3) + 0.19, 0.68 + n * 0.06)
                else:
                    lego_start_pose.p = gymapi.Vec3(0.17 - 0.17 * int(i % 3) + 0.25, 0.11 - 0.11 * int(i / 3) + 0.19, 0.68 + n * 0.06)

                lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.785)
                
                lego_assets.append(lego_asset)
                lego_start_poses.append(lego_start_pose)

        lego_asset_options = gymapi.AssetOptions()
        lego_asset_options.disable_gravity = False
        lego_asset_options.fix_base_link = True
        lego_asset_options.thickness = 0.001
        lego_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # lego_asset_options.density = 2000
        flat_lego_begin = len(lego_assets)        
        ran_list = [0 ,0 ,0, 1, 2, 2]
        lego_list = [0, 5, 6]
        bianchang = [0.03, 0.045, 0.06]        
        for j in range(10):
            random.shuffle(ran_list)
            lego_center = [0.254 - bianchang[ran_list[0]] + 0.25, 0.175 + 0.19 - 0.039 * j, 0.63]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0] , lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[0]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[0]] + bianchang[ran_list[1]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[1]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[1]] + bianchang[ran_list[2]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[2]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[2]] + bianchang[ran_list[3]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[3]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[3]] + bianchang[ran_list[4]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[4]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[4]] + bianchang[ran_list[5]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[5]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)        
        
        flat_lego_end = len(lego_assets)

        extra_lego_asset_options = gymapi.AssetOptions()
        extra_lego_asset_options.disable_gravity = False
        extra_lego_asset_options.fix_base_link = True

        extra_lego_assets = []

        # fake extra lego
        extra_lego_asset = self.gym.load_asset(self.sim, asset_root, "urdf/blender/assets_for_insertion/urdf/12x12x1_real.urdf", extra_lego_asset_options)
        extra_lego_assets.append(extra_lego_asset)

        extra_lego_start_pose = gymapi.Transform()
        extra_lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.0)
        # Assets visualization
        extra_lego_start_pose.p = gymapi.Vec3(0.25, -0.35, 0.618)

        # compute aggregate size
        max_agg_bodies = self.num_arm_hand_bodies + 2 + 1 + len(lego_assets) + 5 + 10 
        max_agg_shapes = self.num_arm_hand_shapes + 2 + 1 + len(lego_assets) + 5 + 100 

        self.arm_hands = []
        self.envs = []

        self.lego_init_states = []
        self.hand_start_states = []
        self.extra_lego_init_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.predict_object_indices = []
        self.table_indices = []
        self.lego_indices = []
        self.lego_segmentation_indices = []
        self.block_half_side_length = []
        self.extra_object_indices = []

        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            arm_hand_actor = self.gym.create_actor(env_ptr, arm_hand_asset, arm_hand_start_pose, "hand", i, 0, 0)
            self.hand_start_states.append([arm_hand_start_pose.p.x,
                                           arm_hand_start_pose.p.y,
                                           arm_hand_start_pose.p.z,
                                           arm_hand_start_pose.r.x,
                                           arm_hand_start_pose.r.y,
                                           arm_hand_start_pose.r.z,
                                           arm_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, arm_hand_actor, arm_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, arm_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.9, 0.8)
            )
            self.table_indices.append(table_idx)
            
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            for object_shape_prop in table_shape_props:
                object_shape_prop.friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            # add box
            for box_i, box_asset in enumerate(box_assets):
                box_handle = self.gym.create_actor(env_ptr, box_asset, box_start_poses[box_i], "box_{}".format(box_i), i, 0, 0)
                self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))

            # add lego
            color_map = [[0.8, 0.64, 0.2], [0.13, 0.54, 0.13], [0, 0.4, 0.8], [1, 0.54, 0], [0.69, 0.13, 0.13], [0.69, 0.13, 0.13], [0, 0.4, 0.8], [0.8, 0.64, 0.2]]
            lego_idx = []
            segmentation_id = i % 8
            self.block_half_side_length.append(lego_g_half_length[segmentation_id])
                
            for lego_i, lego_asset in enumerate(lego_assets):
                lego_handle = self.gym.create_actor(env_ptr, lego_asset, lego_start_poses[lego_i], "lego_{}".format(lego_i), i, 0, lego_i + 1)
                self.lego_init_states.append([lego_start_poses[lego_i].p.x, lego_start_poses[lego_i].p.y, lego_start_poses[lego_i].p.z,
                                            lego_start_poses[lego_i].r.x, lego_start_poses[lego_i].r.y, lego_start_poses[lego_i].r.z, lego_start_poses[lego_i].r.w,
                                            0, 0, 0, 0, 0, 0])
                idx = self.gym.get_actor_index(env_ptr, lego_handle, gymapi.DOMAIN_SIM)
                if lego_i == segmentation_id:
                    self.lego_segmentation_indices.append(idx)

                lego_idx.append(idx)
                lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, lego_handle)
                for lego_body_prop in lego_body_props:
                    if flat_lego_end > lego_i >= flat_lego_begin:
                        lego_body_prop.mass *= 1
                self.gym.set_actor_rigid_body_properties(env_ptr, lego_handle, lego_body_props)

                color = color_map[lego_i % 8]
                if flat_lego_end > lego_i >= flat_lego_begin:
                    color = color_map[random.randint(0, 7)]
                self.gym.set_rigid_body_color(env_ptr, lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))
            self.lego_indices.append(lego_idx)

            extra_lego_handle = self.gym.create_actor(env_ptr, extra_lego_assets[0], extra_lego_start_pose, "extra_lego", i, 0, 0)
            self.extra_lego_init_states.append([extra_lego_start_pose.p.x, extra_lego_start_pose.p.y, extra_lego_start_pose.p.z,
                                        extra_lego_start_pose.r.x, extra_lego_start_pose.r.y, extra_lego_start_pose.r.z, extra_lego_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0])
            self.gym.get_actor_index(env_ptr, extra_lego_handle, gymapi.DOMAIN_SIM)
            extra_object_idx = self.gym.get_actor_index(env_ptr, extra_lego_handle, gymapi.DOMAIN_SIM)
            self.gym.set_rigid_body_color(env_ptr, extra_lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))
            self.extra_object_indices.append(extra_object_idx)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.arm_hands.append(arm_hand_actor)

        # Acquire specific links.
        sensor_handles = [0, 1, 2, 3, 4, 5, 6]
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)

        self.fingertip_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, name) for name in self.fingertip_names]

        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.lego_init_states = to_torch(self.lego_init_states, device=self.device).view(self.num_envs, len(lego_assets), 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.lego_indices = to_torch(self.lego_indices, dtype=torch.long, device=self.device)
        self.lego_segmentation_indices = to_torch(self.lego_segmentation_indices, dtype=torch.long, device=self.device)
        self.block_half_side_length = to_torch(self.block_half_side_length, device=self.device)

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.progress_buf[:] = compute_hand_reward(
            self.reset_buf, self.progress_buf, self.contacts, self.segmentation_target_init_pos,
            self.max_episode_length, self.segmentation_target_pos, self.segmentation_target_rot, self.fingertip_poses,
            self.actions, self.segmentation_target_side_pos
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Construct states buf: 
        self.states_buf[:, 0:19] = unscale(self.arm_hand_dof_pos[:, 0:19],
                                                            self.arm_hand_dof_lower_limits[0:19],
                                                            self.arm_hand_dof_upper_limits[0:19])
        self.states_buf[:, 19:38] = self.vel_obs_scale * self.arm_hand_dof_vel[:, 0:19]

        # Add finger states
        id = 38
        
        self.fingertip_poses = []
        for i in range(5):
            pos = self.rigid_body_states[:, self.fingertip_handles[i], 0:3]
            rot = self.rigid_body_states[:, self.fingertip_handles[i], 3:7]
            fingertip_pos = pos + quat_apply(rot[:], to_torch(self.fingertip_adjustment_params[i][0], device=self.device).repeat(self.num_envs, 1) * self.fingertip_adjustment_params[i][1])
            self.fingertip_poses.append(fingertip_pos)
            self.states_buf[:, id:id+3] = fingertip_pos
            self.states_buf[:, id+3:id+13] = self.rigid_body_states[:, self.fingertip_handles[i], 3:13]
            id += 13

        # Add Link7 state
        self.states_buf[:, id:id+13] = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:13]
        id += 13

        # Add lego target state
        self.states_buf[:, id:id + 13] = self.root_state_tensor[self.lego_segmentation_indices, 0:13]
        
        # Clone states buf to obs buf
        self.obs_buf[:, :] = self.states_buf[:, :]

        self.segmentation_target_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3]
        self.segmentation_target_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7]
        self.segmentation_target_side_pos = []
        for direction in [1, -1]:
            pos = self.segmentation_target_pos + quat_apply(self.segmentation_target_rot[:], to_torch([0,direction,0], device="cuda:0") * self.block_half_side_length.unsqueeze(-1))
            self.segmentation_target_side_pos.append(pos)

        contacts = self.contact_tensor.reshape(self.num_envs, -1, 3)  # 39+27  # TODO
        contacts = contacts[:, self.sensor_handle_indices, :] # 12
        contacts = torch.norm(contacts, dim=-1)
        self.contacts = torch.where(contacts >= 0.1, 1.0, 0.0)

        for i in range(len(self.contacts[0])):
            if self.contacts[0][i] == 1.0:
                self.gym.set_rigid_body_color(
                            self.envs[0], self.hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.3, 0.3))
            else:
                self.gym.set_rigid_body_color(
                            self.envs[0], self.hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))

    def reset_idx(self, env_ids):
        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_hand_dofs * 2 + 5), device=self.device)
        
        lego_init_rand_floats = torch_rand_float(-1.0, 1.0, (self.num_envs * 132, 3), device=self.device)
        lego_init_rand_floats.view(self.num_envs, 132, 3)[:, 72:, :] = 0

        # reset object
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 0:7] = self.lego_init_states[env_ids].view(-1, 13)[:, 0:7].clone()
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 7:13] = torch.zeros_like(self.root_state_tensor[self.lego_indices[env_ids].view(-1), 7:13])
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 0:1] = self.lego_init_states[env_ids].view(-1, 13)[:, 0:1].clone() + lego_init_rand_floats[:, 0].unsqueeze(-1) * 0.02
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 1:2] = self.lego_init_states[env_ids].view(-1, 13)[:, 1:2].clone() + lego_init_rand_floats[:, 1].unsqueeze(-1) * 0.02

        # randomize segmentation object
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 0] = 0.25 + rand_floats[env_ids, 0] * 0.2
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 1] = 0.19 + rand_floats[env_ids, 0] * 0.15
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 2] = 0.9

        lego_ind = self.lego_indices[env_ids].view(-1).to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(lego_ind), len(lego_ind))

        # reset shadow hand
        pos = self.arm_hand_default_dof_pos
        self.arm_hand_dof_pos[env_ids, 0:19] = pos[0:19]
        self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel
        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.post_reset(env_ids, hand_indices)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def post_reset(self, env_ids, hand_indices):
        # step physics and render each frame
        for _ in range(80):
            self.render()
            self.gym.simulate(self.sim)
        
        # self.render_for_camera()
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.segmentation_target_init_pos[env_ids] = self.root_state_tensor[self.lego_segmentation_indices[env_ids], 0:3].clone()
        self.segmentation_target_init_rot[env_ids] = self.root_state_tensor[self.lego_segmentation_indices[env_ids], 3:7].clone()

        print("post_reset finish")

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)

        # Set hand actions
        self.cur_targets[:, self.actuated_hand_dof_indices] = scale(self.actions[:, 7:],
                                                                self.arm_hand_dof_lower_limits[self.actuated_hand_dof_indices],
                                                                self.arm_hand_dof_upper_limits[self.actuated_hand_dof_indices])
        self.cur_targets[:, self.actuated_hand_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                    self.actuated_hand_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_hand_dof_indices]


        # Set arm actions
        self.cur_targets[:, self.actuated_arm_dof_indices] = self.arm_hand_dof_pos[:, self.actuated_arm_dof_indices] + self.actions[:, :7] * self.sim_params.dt * self.dof_speed_scale


        self.cur_targets[:, :] = tensor_clamp(self.cur_targets[:, :],
                                                self.arm_hand_dof_lower_limits[:],
                                                self.arm_hand_dof_upper_limits[:])

        self.prev_targets[:, :] = self.cur_targets[:, :]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_observations()
        self.compute_reward()

    def draw_point(
        self,
        env,
        center,
        rotation,
        ax="xyz",
        radius=0.02,
        num_segments=32,
        color=(1, 0, 0),
    ):
        rotation = rotation.cpu().numpy()
        center = center.cpu().numpy()
        rot_matrix = R.from_quat(rotation).as_matrix()

        for ax in list(ax):
            # 根据指定的轴选择正确的平面
            if ax.lower() == "x":
                plane_axes = [1, 2]  # yz平面
            elif ax.lower() == "y":
                plane_axes = [0, 2]  # xz平面
            else:  # 默认为z轴
                plane_axes = [0, 1]  # xy平面

            points = []
            for i in range(num_segments + 1):
                angle = 2 * math.pi * i / num_segments
                # 在选定的平面上计算点
                local_point = np.zeros(3)
                local_point[plane_axes[0]] = radius * math.cos(angle)
                local_point[plane_axes[1]] = radius * math.sin(angle)

                # 将局部坐标转换为全局坐标
                global_point = center + rot_matrix @ local_point
                points.append(global_point)

            for i in range(num_segments):
                start = points[i]
                end = points[i + 1]
                self.gym.add_lines(self.viewer, env, 1, [*start, *end], color)

    def step(self, actions):

        # apply actions
        self.pre_physics_step(actions)

        self.gym.clear_lines(self.viewer)

        self.draw_point(self.envs[0], self.root_state_tensor[self.lego_segmentation_indices[0], 0:3], self.root_state_tensor[self.lego_segmentation_indices[0], 3:7], ax="xyz", radius=0.03, num_segments=32, color=(1, 0, 0))
        self.draw_point(self.envs[0], self.segmentation_target_init_pos[0], self.segmentation_target_init_rot[0], ax="xyz", radius=0.03, num_segments=32, color=(0, 1, 0))

        # try:
        #     for env_id in range(min(4, self.num_envs)):
        #         # Draw point at target position
        #         self.draw_point(self.envs[env_id], self.root_state_tensor[self.lego_segmentation_indices[env_id], 0:3], self.root_state_tensor[self.lego_segmentation_indices[env_id], 3:7], ax="xyz", radius=0.03, num_segments=32, color=(1, 0, 0))
        #         # Draw points at fingertips
        #         for finger_i in range(5):
        #             self.draw_point(self.envs[env_id], self.fingertip_poses[finger_i], torch.tensor([0,0,0,1], dtype=torch.float, device="cuda:0"), ax="xyz", radius=0.03, num_segments=32, color=(0, 1, 0))
        # except:
        #     pass

        # self.draw_point(self.envs[1], self.root_state_tensor[self.lego_segmentation_indices[1], 0:3], self.root_state_tensor[self.lego_segmentation_indices[1], 3:7], ax="xyz", radius=0.03, num_segments=32, color=(1, 0, 0))

        # for env in self.envs:
        #     draw_line(gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 1), gymapi.Vec3(0, 0, 1), self.gym, self.viewer, env)
        #     draw_line(gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 1, 0), gymapi.Vec3(0, 1, 0), self.gym, self.viewer, env)
        #     draw_line(gymapi.Vec3(0, 0, 0), gymapi.Vec3(1, 0, 0), gymapi.Vec3(1, 0, 0), self.gym, self.viewer, env)

        # step physics and render each frame
        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

    def get_states(self):
        return self.states_buf

    def render(self):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()          

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)



#####################################################################
###=========================jit functions=========================###
#####################################################################

def compute_hand_reward(
    reset_buf, progress_buf, arm_contacts, segmentation_target_init_pos,
    max_episode_length: float, segmentation_target_pos, segmentation_target_rot, fingertip_poses, actions, segmentation_target_side_pos
):
    # Reward for grasping
    grasp_distance_two_directions = [0, 0]
    for direction in range(2):
        for i in range(5):
            grasp_distance_two_directions[direction] += torch.norm(segmentation_target_side_pos[direction] - fingertip_poses[i], p=2, dim=-1)
    grasp_reward = torch.exp(-torch.min(grasp_distance_two_directions[0], grasp_distance_two_directions[1]))
    # dist_rew = torch.clamp(- 0.2 *arm_hand_finger_dist, None, -0.06)

    # Reward for block rotation
    # euler = quat_to_euler_xyz(segmentation_target_rot)
    # target_rot_reward = (-torch.abs(euler[:, 0]) - torch.abs(euler[:, 1])) * 100
    target_rot_reward = 0

    # Reward for hand rotation


    # Reward for object lifting
    # object_up_reward = torch.clamp(segmentation_target_pos[:, 2]-segmentation_target_init_pos[:, 2], min=0, max=0.1) * 2000 - torch.clamp(segmentation_target_pos[:, 0]-segmentation_target_init_pos[:, 0], min=0, max=0.1) * 2000 - torch.clamp(segmentation_target_pos[:, 1]-segmentation_target_init_pos[:, 1], min=0, max=0.1) * 2000
    object_up_reward = 50 * torch.clamp(segmentation_target_pos[:, 2] - segmentation_target_init_pos[:, 2], min=0, max=None)
    # object_up_reward = 0

    # action_penalty = torch.sum(actions ** 2, dim=-1) * 0.005
    action_penalty = 0

    # arm_contacts_penalty = torch.sum(arm_contacts, dim=-1) * 50
    arm_contacts_penalty = 0

    
    reward = grasp_reward + target_rot_reward + object_up_reward

    print(f"Total reward {reward.mean().item():.2f}, grasp reward {grasp_reward.mean().item():.2f}, target rot reward {target_rot_reward:.2f}, object_up_reward {object_up_reward.mean().item():.2f}")

    # Fall penalty: distance to the goal is larger than a threshold
    # Check env termination conditions, including maximum success number
    resets = reset_buf

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    return reward, resets, progress_buf

@torch.jit.script
def quat_to_euler_xyz(quat):
    """
    Convert quaternions to Euler angles (roll, pitch, yaw).
    :param quat: quaternions with shape (..., 4)
    :return: tuple of roll, pitch, yaw with shape (..., 3)
    """
    # Ensure quat has shape (..., 4)
    assert quat.shape[-1] == 4

    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    # Extract quaternion components
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.pi / 2,
        torch.asin(sinp)
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)