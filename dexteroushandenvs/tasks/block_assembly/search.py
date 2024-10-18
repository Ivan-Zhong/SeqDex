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

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R
import warnings


class Search:

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
        # self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.fingertip_names = ["R_thumb_distal", "R_index_distal", "R_middle_distal", "R_ring_distal", "R_pinky_distal"]
        self.fingertip_adjustment_params = [[[0, 0.5, 0.1], 0.04], [[0.18, 0.9, 0.1], 0.04], [[0.15, 0.9, 0.1], 0.04], [[0.2, 0.9, 0.1], 0.04], [[0.2, 0.8, 0.1], 0.04]]

        self.cfg["env"]["numObservations"] = 154
        self.cfg["env"]["numStates"] = 154
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

        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

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
            cam_pos = gymapi.Vec3(1, -0.1, 1.5)
            cam_target = gymapi.Vec3(-0.7, -0.1, 0.5)
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
        self.arm_hand_default_dof_pos = to_torch([0.0, 0.45, 0.0, 1.78, 0.0, -0.5, -1.571, 0.0, 0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0], dtype=torch.float, device=self.device)

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
        self.delta_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.E_prev = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        self.base_pos = self.rigid_body_states[:, 0, 0:3].clone()
        self.segmentation_target_init_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3].clone()
        self.segmentation_target_init_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7].clone()
        self.last_all_lego_pos = self.root_state_tensor[self.lego_indices, 0:3].clone()  # (num_envs, 132, 3)
        self.now_all_lego_pos = self.root_state_tensor[self.lego_indices, 0:3].clone()  # (num_envs, 132, 3)

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
        actuated_dof_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "R_thumb_MCP_joint1", "R_thumb_MCP_joint2", "R_index_MCP_joint", "R_middle_MCP_joint", "R_ring_MCP_joint", "R_pinky_MCP_joint"]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(arm_hand_asset, name) for name in actuated_dof_names]
        actuated_hand_dof_names = ["R_thumb_MCP_joint1", "R_thumb_MCP_joint2", "R_index_MCP_joint", "R_middle_MCP_joint", "R_ring_MCP_joint", "R_pinky_MCP_joint"]
        self.actuated_hand_dof_indices = [self.gym.find_asset_dof_index(arm_hand_asset, name) for name in actuated_hand_dof_names]
        actuated_arm_dof_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.actuated_arm_dof_indices = [self.gym.find_asset_dof_index(arm_hand_asset, name) for name in actuated_arm_dof_names]

        arm_hand_dof_props = self.gym.get_asset_dof_properties(arm_hand_asset)
        arm_hand_dof_props["driveMode"][:] = gymapi.DOF_MODE_POS

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.arm_hand_dof_default_vel = []

        arm_hand_dof_lower_limits_list = [-3.1, -2.268, -3.1, -2.355, -3.1, -2.233, -6.28]
        arm_hand_dof_upper_limits_list = [3.1, 2.268, 3.1, 2.355, 3.1, 2.233, 6.28]
        
        arm_hand_dof_stiffness_list = [200, 200, 100, 100, 50, 50, 50]
        arm_hand_dof_damping_list = [20, 20, 10, 10, 10, 5, 5]
        arm_hand_dof_effort_list = [60, 60, 30, 30, 10, 10, 10]
        arm_hand_dof_velocity_list = [1, 1, 1, 1, 1, 1, 1]

        for i in range(self.num_arm_hand_dofs):
            if i < 7:
                self.arm_hand_dof_lower_limits.append(arm_hand_dof_lower_limits_list[i])
                self.arm_hand_dof_upper_limits.append(arm_hand_dof_upper_limits_list[i])
            else:
                self.arm_hand_dof_lower_limits.append(arm_hand_dof_props['lower'][i])
                self.arm_hand_dof_upper_limits.append(arm_hand_dof_props['upper'][i])
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
        self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)

        # Put objects in the scene.
        arm_hand_start_pose = gymapi.Transform()
        arm_hand_start_pose.p = gymapi.Vec3(-0.3, 0.0, 0.6)
        arm_hand_start_pose.r = gymapi.Quat(0, 0, 1, 0)

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
        table_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.0)

        # create box asset
        box_assets = []
        box_start_poses = []

        box_thin = 0.01
        box_xyz = [0.60, 0.416, 0.165]
        box_offset = [0.25, 0.19, 0]
        self.box_bounds_x = [(box_offset[0] + box_xyz[0] / 2 - box_thin, 1), (box_offset[0] - box_xyz[0] / 2 + box_thin, -1)]
        self.box_bounds_y = [(box_offset[1] + box_xyz[1] / 2 - box_thin, 1), (box_offset[1] - box_xyz[1] / 2 + box_thin, -1)]

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
        all_lego_files_name = ['1x2.urdf', '1x2_curve.urdf', '1x3_curve_soft.urdf', '1x3_curve.urdf', '1x1.urdf', '1x3.urdf', '1x4.urdf', '2x2_curve_soft.urdf']

        lego_assets = []
        lego_start_poses = []

        lego_asset_options = gymapi.AssetOptions()
        lego_asset_options.disable_gravity = False
        lego_asset_options.thickness = 0.00001
        lego_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # lego_asset_options.density = 300.0
        for n in range(16):
            for i, lego_file_name in enumerate(all_lego_files_name):
                lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)

                lego_start_pose = gymapi.Transform()
                if n % 2 == 0:
                    lego_start_pose.p = gymapi.Vec3(-0.17 + 0.17 * int(i % 3) + 0.25, -0.11 + 0.11 * int(i / 3) + 0.19, 0.68 + n * 0.06)
                else:
                    lego_start_pose.p = gymapi.Vec3(0.17 - 0.17 * int(i % 3) + 0.25, 0.11 - 0.11 * int(i / 3) + 0.19, 0.68 + n * 0.06)

                lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.785)
                
                lego_assets.append(lego_asset)
                lego_start_poses.append(lego_start_pose)

        # Lego board
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

        # Create actors
        self.arm_hands = []
        self.envs = []
        self.lego_init_states = []
        self.hand_start_states = []
        self.hand_indices = []
        self.lego_indices = []
        self.lego_segmentation_indices = []
        self.segmentation_types = []

        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # if self.aggregate_mode >= 1:
            #     self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

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

            arm_hand_actor_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, arm_hand_actor)
            for _, arm_hand_actor_shape_prop in enumerate(arm_hand_actor_shape_props):
                arm_hand_actor_shape_prop.friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, arm_hand_actor, arm_hand_actor_shape_props)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.9, 0.8)
            )
            
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            # for object_shape_prop in table_shape_props:
            #     object_shape_prop.friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            # add box
            for box_i, box_asset in enumerate(box_assets):
                box_handle = self.gym.create_actor(env_ptr, box_asset, box_start_poses[box_i], "box_{}".format(box_i), i, 0, 0)
                self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))

            # add lego
            color_map = [[0.8, 0.64, 0.2], [0.13, 0.54, 0.13], [0, 0.4, 0.8], [1, 0.54, 0], [0.69, 0.13, 0.13], [0.69, 0.13, 0.13], [0, 0.4, 0.8], [0.8, 0.64, 0.2]]
            lego_idx = []
            segmentation_id = i % 8
            segmentation_type = torch.zeros((1, 8))
            segmentation_type[0, segmentation_id] = 1
            self.segmentation_types.append(segmentation_type)

            for lego_i, lego_asset in enumerate(lego_assets):
                lego_handle = self.gym.create_actor(env_ptr, lego_asset, lego_start_poses[lego_i], "lego_{}".format(lego_i), i, 0, lego_i + 1)
                self.lego_init_states.append([lego_start_poses[lego_i].p.x, lego_start_poses[lego_i].p.y, lego_start_poses[lego_i].p.z,
                                            lego_start_poses[lego_i].r.x, lego_start_poses[lego_i].r.y, lego_start_poses[lego_i].r.z, lego_start_poses[lego_i].r.w,
                                            0, 0, 0, 0, 0, 0])
                idx = self.gym.get_actor_index(env_ptr, lego_handle, gymapi.DOMAIN_SIM)
                if lego_i == segmentation_id:
                    self.lego_segmentation_indices.append(idx)

                lego_idx.append(idx)

                color = color_map[lego_i % 8]
                self.gym.set_rigid_body_color(env_ptr, lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))
            self.lego_indices.append(lego_idx)

            extra_lego_handle = self.gym.create_actor(env_ptr, extra_lego_assets[0], extra_lego_start_pose, "extra_lego", i, 0, 0)
            self.gym.set_rigid_body_color(env_ptr, extra_lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))

            # if self.aggregate_mode > 0:
            #     self.gym.end_aggregate(env_ptr)

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
        self.segmentation_types = torch.concat(self.segmentation_types, dim=0)

    def compute_reward(self):

        fingers_pos = [self.finger_thumb_pos, self.finger_index_pos, self.finger_middle_pos]
        fingers_names = ["thumb", "index", "middle"]
        fingers_weights = [2, 1, 1]
        distance_reward = 0
        for i in range(len(fingers_pos)):
            finger_pos = fingers_pos[i]
            finger_weight = fingers_weights[i]
            finger_name = fingers_names[i]
            finger_dist = torch.norm(self.seg_lego_pos - finger_pos, p=2, dim=-1)
            print(f"{finger_name} dist:", finger_dist[0])
            distance_reward += finger_weight * 10 * torch.exp(- 4 * torch.clamp(finger_dist - 0.15, 0, None))

        grasp_fingers_pos = [
            self.middle_point
        ]
        pose_dist = sum([tolerance(point_, self.seg_lego_pos, 0.15, 0.01) for point_ in grasp_fingers_pos]) / len(grasp_fingers_pos)
        print("Pose dist:", pose_dist[0])
        pose_reward = pose_dist * 10

        cover_penalty = self.cover_penalty.squeeze(-1) * 400

        # action_reward = torch.norm(self.delta_targets, p=2, dim=-1) * 2

        # block_movement_reward = self.block_movement_reward.squeeze(-1) * 1000

        # block_linvel_reward = self.block_linvel_reward.squeeze(-1) * 0.05

        # block_angvel_reward = self.block_angvel_reward.squeeze(-1) * 0.005

        lift_reward = (1 + pose_dist) * 400 * torch.clamp((self.seg_lego_pos[:, 2] - self.segmentation_target_init_pos[:, 2]), 0, None)

        # total_reward = distance_reward + pose_reward + cover_penalty + action_reward + block_movement_reward
        total_reward = distance_reward + pose_reward + cover_penalty + lift_reward
        # total_reward = distance_reward + pose_reward + cover_penalty + action_reward + block_movement_reward + block_linvel_reward + block_angvel_reward + lift_reward - self.E_prev

        # self.E_prev = distance_reward + pose_reward + lift_reward + angle_reward
        # self.E_prev = distance_reward + pose_reward + cover_penalty + action_reward + block_movement_reward + block_linvel_reward + block_angvel_reward + lift_reward

        # Print all reward in a good format

        print("#####################################")
        print(f"Distance reward {distance_reward.mean().item():.2f}")
        print(f"Pose reward {pose_reward.mean().item():.2f}")
        print(f"Cover penalty {cover_penalty.mean().item():.2f}")
        # print(f"Action reward {action_reward.mean().item():.2f}")
        # print(f"Block movement reward {block_movement_reward.mean().item():.2f}")
        # print(f"Block linvel reward {block_linvel_reward.mean().item():.2f}")
        # print(f"Block angvel reward {block_angvel_reward.mean().item():.2f}")
        print(f"Lift reward {lift_reward.mean().item():.2f}")
        print(f"Total reward {total_reward.mean().item():.2f}")
        print("#####################################")

        # Fall penalty: distance to the goal is larger than a threshold
        # Check env termination conditions, including maximum success number
        resets = self.reset_buf

        timed_out = self.progress_buf >= self.max_episode_length - 1
        resets = torch.where(timed_out, torch.ones_like(resets), resets)
        self.reset_buf[:] = resets
        self.rew_buf[:] = total_reward


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
        # self.states_buf[:, 19:38] = self.arm_hand_dof_vel[:, 0:19]

        # Add finger states
        id = 38
        self.finger_thumb_pos = self.rigid_body_states[:, self.fingertip_handles[0], 0:3]
        self.finger_thumb_rot = self.rigid_body_states[:, self.fingertip_handles[0], 3:7]
        self.finger_index_pos = self.rigid_body_states[:, self.fingertip_handles[1], 0:3]
        self.finger_index_rot = self.rigid_body_states[:, self.fingertip_handles[1], 3:7]
        self.finger_middle_pos = self.rigid_body_states[:, self.fingertip_handles[2], 0:3]
        self.finger_middle_rot = self.rigid_body_states[:, self.fingertip_handles[2], 3:7]
        self.finger_ring_pos = self.rigid_body_states[:, self.fingertip_handles[3], 0:3]
        self.finger_ring_rot = self.rigid_body_states[:, self.fingertip_handles[3], 3:7]
        self.finger_pinky_pos = self.rigid_body_states[:, self.fingertip_handles[4], 0:3]
        self.finger_pinky_rot = self.rigid_body_states[:, self.fingertip_handles[4], 3:7]

        self.finger_thumb_pos += quat_apply(self.finger_thumb_rot[:], to_torch([0, 0.5, 0.1], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.finger_index_pos += quat_apply(self.finger_index_rot[:], to_torch([0.18, 0.9, 0.1], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.finger_middle_pos+= quat_apply(self.finger_middle_rot[:], to_torch([0.15, 0.9, 0.1],device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.finger_ring_pos  += quat_apply(self.finger_ring_rot[:], to_torch([0.2, 0.9, 0.1],  device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.finger_pinky_pos += quat_apply(self.finger_pinky_rot[:], to_torch([0.2, 0.8, 0.1], device=self.device).repeat(self.num_envs, 1) * 0.04)

        self.seg_lego_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3]

        self.states_buf[:, id+0:id+3] = self.finger_thumb_pos - self.seg_lego_pos 
        self.states_buf[:, id+3:id+6] = self.finger_index_pos - self.seg_lego_pos
        self.states_buf[:, id+6:id+9] = self.finger_middle_pos - self.seg_lego_pos 
        self.states_buf[:, id+9:id+12] = self.finger_ring_pos - self.seg_lego_pos 
        self.states_buf[:, id+12:id+15] = self.finger_pinky_pos - self.seg_lego_pos 
        self.middle_point = (self.finger_thumb_pos + self.finger_index_pos + self.finger_middle_pos) / 3
        self.states_buf[:, id+15:id+18] = self.middle_point - self.seg_lego_pos
        id += 6 * 3

        for i in range(5):
            self.states_buf[:, id:id+10] = self.rigid_body_states[:, self.fingertip_handles[i], 3:13]
            id += 10

        self.states_buf[:, id:id+13] = self.actions
        id += 13

        self.states_buf[:, id:id+13] = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:13]
        id += 13

        self.states_buf[:, id:id + 13] = self.root_state_tensor[self.lego_segmentation_indices, 0:13]
        id += 13

        self.states_buf[:, id:id + 8] = self.segmentation_types
        id += 8

        self.now_all_lego_pos = self.root_state_tensor[self.lego_indices, 0:3]  # (num_envs, 132, 3)
        target_pos = self.seg_lego_pos.unsqueeze(1)
        condition = (torch.abs(self.now_all_lego_pos[:, :, :2] - target_pos[:, :, :2]) < 0.06).all(dim=2) \
                    & (self.now_all_lego_pos[:, :, 2] > (target_pos[:, :, 2] + 1e-5))  # (num_envs, 132)
        
        self.is_covered = torch.any(condition, dim=1, keepdim=True).float()
        self.states_buf[:, id:id + 1] = self.is_covered
        id += 1

        legos_dist = torch.norm(self.now_all_lego_pos - target_pos, p=2, dim=2)  # (num_envs, 132)
        cover_penalty_per_block = torch.where(condition, legos_dist - 0.2, 0.0)  # (num_envs, 132)
        self.cover_penalty = torch.sum(cover_penalty_per_block, dim=1, keepdim=True)  # (num_envs, 1)

        legos_movement = torch.norm(self.now_all_lego_pos - self.last_all_lego_pos, p=2, dim=2)  # (num_envs, 132)
        legos_movement_reward_per_block = torch.where(condition, legos_movement - 0.03, legos_movement * 0.2)  # (num_envs, 132)
        self.block_movement_reward = torch.sum(legos_movement_reward_per_block, dim=1, keepdim=True)  # (num_envs, 1)

        self.now_all_lego_linvel = self.root_state_tensor[self.lego_indices, 7:10]  # (num_envs, 132, 3)
        self.now_all_lego_angvel = self.root_state_tensor[self.lego_indices, 10:13]  # (num_envs, 132, 3)
        now_all_lego_linvel_norm = torch.norm(self.now_all_lego_linvel, p=2, dim=2)  # (num_envs, 132)
        now_all_lego_angvel_norm = torch.norm(self.now_all_lego_angvel, p=2, dim=2)  # (num_envs, 132)
        print("now_all_lego_linvel_norm max:", now_all_lego_linvel_norm.max().item())
        print("now_all_lego_angvel_norm max:", now_all_lego_angvel_norm.max().item())
        # linvel_reward_per_block = torch.where(condition, now_all_lego_linvel_norm, now_all_lego_linvel_norm * 0.1)  # (num_envs, 132)
        # angvel_reward_per_block = torch.where(condition, now_all_lego_angvel_norm, now_all_lego_angvel_norm * 0.1)  # (num_envs, 132)
        self.block_linvel_reward = torch.sum(now_all_lego_linvel_norm, dim=1, keepdim=True)
        self.block_angvel_reward = torch.sum(now_all_lego_angvel_norm, dim=1, keepdim=True)


        self.last_all_lego_pos[:, :, :] = self.now_all_lego_pos[:, :, :]

        # Clone states buf to obs buf
        self.obs_buf = self.states_buf


        # contacts = self.contact_tensor.reshape(self.num_envs, -1, 3)  # 39+27  # TODO
        # contacts = contacts[:, self.sensor_handle_indices, :] # 12
        # contacts = torch.norm(contacts, dim=-1)
        # self.contacts = torch.where(contacts >= 0.1, 1.0, 0.0)

        # for i in range(len(self.contacts[0])):
        #     if self.contacts[0][i] == 1.0:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.3, 0.3))
        #     else:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))

    def reset_idx(self, env_ids):
        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        
        lego_init_rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids) * 128, 3), device=self.device)
        # lego_init_rand_floats.view(self.num_envs, 132, 3)[:, 72:, :] = 0

        # reset object
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 0:7] = self.lego_init_states[env_ids].view(-1, 13)[:, 0:7].clone()
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 7:13] = torch.zeros_like(self.root_state_tensor[self.lego_indices[env_ids].view(-1), 7:13])
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 0:1] = self.lego_init_states[env_ids].view(-1, 13)[:, 0:1].clone() + lego_init_rand_floats[:, 0].unsqueeze(-1) * 0.02
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 1:2] = self.lego_init_states[env_ids].view(-1, 13)[:, 1:2].clone() + lego_init_rand_floats[:, 1].unsqueeze(-1) * 0.02

        # randomize segmentation object
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 0] = 0.25 + rand_floats[env_ids, 0] * 0.2
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 1] = 0.19 + rand_floats[env_ids, 1] * 0.15
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

        self.last_all_lego_pos[env_ids] = self.root_state_tensor[self.lego_indices[env_ids], 0:3].clone()
        self.now_all_lego_pos[env_ids] = self.root_state_tensor[self.lego_indices[env_ids], 0:3].clone()

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

        self.delta_targets[:, :] = self.cur_targets[:, :] - self.prev_targets[:, :]

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


_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x, value_at_1, sigmoid):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
        x: A scalar or PyTorch tensor of shape (batch_size, 1).
        value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
        sigmoid: String, choice of sigmoid type.

    Returns:
        A PyTorch tensor with values between 0.0 and 1.0.

    Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
          `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                             'got {}.'.format(value_at_1))
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                             'got {}.'.format(value_at_1))

    if sigmoid == 'gaussian':
        scale = torch.sqrt(-2 * torch.log(torch.tensor(value_at_1)))
        return torch.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == 'hyperbolic':
        scale = torch.acosh(1 / torch.tensor(value_at_1))
        return 1 / torch.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = torch.sqrt(1 / torch.tensor(value_at_1) - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / torch.tensor(value_at_1) - 1
        return 1 / (torch.abs(x) * scale + 1)

    elif sigmoid == 'cosine':
        scale = torch.acos(2 * torch.tensor(value_at_1) - 1) / torch.pi
        scaled_x = x * scale
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore', message='invalid value encountered in cos')
            cos_pi_scaled_x = torch.cos(torch.pi * scaled_x)
        return torch.where(torch.abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, torch.tensor(0.0))

    elif sigmoid == 'linear':
        scale = 1 - torch.tensor(value_at_1)
        scaled_x = x * scale
        return torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x, torch.tensor(0.0))

    elif sigmoid == 'quadratic':
        scale = torch.sqrt(1 - torch.tensor(value_at_1))
        scaled_x = x * scale
        return torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x ** 2, torch.tensor(0.0))

    elif sigmoid == 'tanh_squared':
        scale = torch.atanh(torch.sqrt(1 - torch.tensor(value_at_1)))
        return 1 - torch.tanh(x * scale) ** 2

    else:
        raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))

def tolerance(x, y, r, margin=0.0, sigmoid='gaussian', value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    """Returns 1 when `x` falls inside the circle centered at `p` with radius `r`, between 0 and 1 otherwise.

    Args:
        x: A batch_size x 3 numpy array representing the points to check.
        y: A length-3 numpy array representing the center of the circle.
        r: Float. The radius of the circle.
        margin: Float. Parameter that controls how steeply the output decreases as `x` moves out-of-bounds.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian', 'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when the distance from `x` to the nearest bound is equal to `margin`. Ignored if `margin == 0`.

    Returns:
        A numpy array with values between 0.0 and 1.0 for each point in the batch.

    Raises:
        ValueError: If `margin` is negative.
    """
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    # Calculate the Euclidean distance from each point in x to p
    distance = torch.norm(x - y, p=2, dim=-1)

    in_bounds = distance <= r
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        d = (distance - r) / margin
        
        value = torch.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return value

def compute_angle_line_plane(p1, p2, plane_normal):
    # Compute the direction vector of the line
    line_direction = p2 - p1  # (batch, 3)

    # Normalize the line direction and the plane normal
    line_direction_normalized = line_direction / torch.norm(line_direction, dim=-1, keepdim=True)  # (batch, 3)
    plane_normal_normalized = plane_normal / torch.norm(plane_normal, dim=-1, keepdim=True)  # (batch, 3)

    # Compute the dot product between the line direction and the plane normal
    dot_product = torch.bmm(line_direction_normalized.unsqueeze(1), plane_normal_normalized.unsqueeze(2)).squeeze()  # (batch)

    # Clamp the dot product to avoid numerical issues with acos
    dot_product_clamped = torch.clamp(dot_product, -1.0, 1.0)

    # Compute the angle between the line direction and the plane normal
    angle_with_normal = torch.acos(dot_product_clamped)  # (batch)

    # Compute the angle between the line and the plane
    angle_line_plane = torch.pi / 2 - angle_with_normal  # (batch)

    return angle_line_plane