import os
import time
import json
import uuid
import random
import math
import sys

import torch
from torch.utils.data import IterableDataset, Dataset
from torch import nn
from torch.nn import functional as f

from torchvision.transforms import ToPILImage

import numpy as np
from pathlib import Path
from rich.progress import Progress

import MalmoPython
import malmoutils
from lxml import etree

import networkx as nx
from skimage.morphology import dilation


torch.multiprocessing.set_sharing_strategy("file_system")

malmoutils.fix_print()


class EnvironmentGenerator(IterableDataset):
    def __init__(
        self, fn, port, batch_size=128, dataset_size=None, steps=50, tic_duration=0.008
    ):
        super().__init__()
        self.tree = etree.parse(fn)
        self.batch_size = batch_size
        self.agent_host = MalmoPython.AgentHost()
        self.dataset_size = dataset_size
        self.current_samples = 0
        self.steps = steps
        self.tic_duration = tic_duration
        self.tolerance = 0.001
        self.render_tolerance = 0.8

        # Load environment
        self.env = MalmoPython.MissionSpec(etree.tostring(self.tree), True)

        # Do not record anything
        self.record = MalmoPython.MissionRecordSpec()

        # Initialize client pool
        pool = MalmoPython.ClientPool()
        info = MalmoPython.ClientInfo("localhost", port)
        pool.add(info)
        experiment_id = str(uuid.uuid1())

        # Initialize environment
        self.agent_host.startMission(self.env, pool, self.record, 0, experiment_id)

        # Loop until the mission starts
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        world_state = self.wait_initial_state()
        frame = world_state.video_frames[-1]
        self.HWC = (frame.height, frame.width, frame.channels)

        print(self.prev_x, self.prev_y, self.prev_z)

        self.agent_host.sendCommand("tp 8.5 4.0 2.5")
        self.expected_x = 8.5
        self.expected_y = 4.0
        self.expected_z = 2.5
        self.expected_yaw = 0
        self.require_yaw_change = True  # ?
        self.require_move = True

        time.sleep(0.1)
        self.prev_state = self.wait_next_state()
        print("got first state")

        self.path = self.generate_path()
        self.start_time = time.time()
        self.best_index = 0

    def init_pathfinding(self):
        # Get the grid
        grid = [
            block == "air"
            for block in json.loads(self.prev_state.observations[-1].text)["board"]
        ]
        grid = ~np.array(grid).reshape((66, 41))
        grid = np.flip(grid, axis=1)
        grid = dilation(grid)

        # Build the graph
        G = nx.grid_graph(dim=grid.shape)

        H, W = grid.shape

        edges = []
        for n in G.nodes:
            if (
                n[0] > 0
                and n[1] > 0
                and (~grid[n[1] - 1 : n[1] + 2, n[0] - 1 : n[0] + 2]).all()
            ):
                edges += [(n, (n[0] - 1, n[1] - 1))]
                edges += [((n[0] - 1, n[1] - 1), n)]
            if (
                n[0] > 0
                and n[1] < H - 1
                and (~grid[n[1] - 1 : n[1] + 2, n[0] - 1 : n[0] + 2]).all()
            ):
                edges += [(n, (n[0] - 1, n[1] + 1))]
                edges += [((n[0] - 1, n[1] + 1), n)]

        G.add_edges_from(edges)

        blocks = []
        for n in G.nodes:
            j, i = n
            if grid[i, j]:
                blocks += [n]

        G.remove_nodes_from(blocks)
        self.G = G

    def find_path(self, goal):
        x_i = self.curr_x
        z_i = self.curr_z

        i_i = z_i + 29.5
        j_i = -x_i + 19.5

        start = (j_i, i_i)

        def dist(a, b):
            (x1, y1) = a
            (x2, y2) = b
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        path = nx.astar_path(self.G, start, goal, heuristic=dist, weight="cost")
        path = [(-j + 19.5, i - 29.5) for j, i in path]
        path = np.array(list(zip(*path)))

        return path

    def follow_path(self, path, alpha=1, beta=0.6):
        print("follow path")
        x = self.curr_x
        z = self.curr_z
        direction = self.curr_yaw

        distance = np.sqrt((path[0, :] - x) ** 2 + (path[1, :] - z) ** 2)
        goal_index = np.argmin(distance) + 1

        if goal_index >= len(distance):
            self.path = self.generate_path()
            self.best_index = 0
            self.start_time = time.time()

            return None, False

        print('goal index', goal_index)
        print('best index', self.best_index)
        
        if goal_index > self.best_index:
            self.start_time = time.time()
            self.best_index = goal_index

        ntics = (time.time() - self.start_time) / self.tic_duration
        if ntics > 600000:  # why 240?
            print("time diff too much", time.time() - self.start_time)
            self.agent_host.sendCommand("tp 8.5 4.0 2.5")
            self.agent_host.sendCommand("setYaw 0")
            self.expected_x = 8.5
            self.expected_y = 4.0
            self.expected_z = 2.5
            self.expected_yaw = 0
            self.require_yaw_change = self.curr_yaw != 0
            self.require_move = self.curr_x != 8.5 or self.curr_z != 2.5
            self.prev_state = self.wait_next_state()
            self.path = self.generate_path()
            self.best_index = 0
            self.start_time = time.time()

            return None, True

        goal = path[:, goal_index]
        print('goal', goal)

        def get_angle(x, z):
            return np.arctan2(-x, z) / np.pi * 180

        target_direction = get_angle(goal[0] - x, goal[1] - z)
        
        print('target dir', target_direction)

        angle_diff = np.mod(target_direction, 360) - np.mod(direction, 360)
        while np.abs(angle_diff) > 180:
            if angle_diff > 180:
                angle_diff += -360
            else:
                angle_diff += 360

        # maybe just start with applying the acceleration only to
        # ang_vel = alpha * angle_diff / 180 * acceleration_factor
        # speed = beta * np.clip((1 - np.abs(ang_vel)) * distance[goal_index] * 0.3 * acceleration_factor, 0, 1)

        ang_vel = alpha * angle_diff / 180
        speed = beta * np.clip((1 - ang_vel) * distance[goal_index] * 0.3, 0, 1)

        velocity = {"speed": speed, "ang_vel": ang_vel}

        return velocity, False

    def generate_path(self):
        print("gen path")
        x_i = self.curr_x
        z_i = self.curr_z
        direction = self.curr_yaw

        i_i = z_i + 29.5
        j_i = -x_i + 19.5

        start = (int(j_i), int(i_i))

        if not start in self.G.nodes:
            print("no start, moving")
            self.agent_host.sendCommand("tp 8.5 4.0 2.5")
            self.expected_x = 8.5
            self.expected_y = 4.0
            self.expected_z = 2.5
            self.expected_yaw = 0
            self.require_yaw_change = self.curr_yaw != 0
            self.require_move = self.curr_x != 8.5 or self.curr_z != 2.5
            self.prev_state = self.wait_next_state()

            x_i = self.curr_x
            z_i = self.curr_y
            direction = self.curr_yaw

            i_i = z_i + 29.5
            j_i = -x_i + 19.5

            start = (int(j_i), int(i_i))

        def dist(a, b):
            (x1, y1) = a
            (x2, y2) = b
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

        while True:
            goal = (np.random.randint(0, 41), np.random.randint(0, 65))
            try:
                path_ji = nx.astar_path(
                    self.G, start, goal, heuristic=dist, weight="cost"
                )
            except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound) as e:
                continue
            break
        path = [(-j + 19.5, i - 29.5) for j, i in path_ji]  # (z, x)
        path = np.array(list(zip(*path)))

        return path

    # examining visual content/object content etc in the earlier conv layers
    # autoencoder lateral part of entorhinal cortex, dorsal is path integration
    # also taking a look at encoding of proximity to objects versus if the objects are within view
    # similarity to cscg
    # cory at ucsd
    # gaze-positioning paper in nature

    def __iter__(self):
        return self

    def __next__(self):
        # Initialize array with max sequence length
        print("next")
        H, W, C = self.HWC
        L = self.steps
        inputs = np.empty((L, H, W, C), dtype=np.uint8)
        actions = np.empty((L, 2), np.float32)
        state = np.empty((L, 3), dtype=np.float32)

        for idx in range(L):
            #print(idx)

            def btoa(pixels):
                return np.reshape(np.frombuffer(pixels, dtype=np.uint8), self.HWC)

            # Fill batch
            pixels = self.prev_state.video_frames[-1].pixels

            inputs[idx] = btoa(pixels).copy()

            x_i = self.curr_x
            z_i = self.curr_z
            direction = self.curr_yaw

            i_i = z_i + 29.5
            j_i = -x_i + 19.5

            start = (int(j_i), int(i_i))

            state[idx] = np.array([x_i, z_i, direction], dtype=np.float32)

            velocity, stuck = self.follow_path(self.path)
            if not velocity:
                if stuck:
                    print("stuck")
                    # return (inputs, actions, state)
                    return None

                # check if position is feasible
                if not self.G.has_node(start):
                    print("no start in next")
                    self.agent_host.sendCommand("tp 8.5 4.0 2.5")
                    self.expected_x = 8.5
                    self.expected_y = 4.0
                    self.expected_z = 2.5
                    self.expected_yaw = 0
                    self.require_yaw_change = self.curr_yaw != 0
                    self.require_move = self.curr_x != 8.5 or self.curr_z != 2.5
                    self.prev_state = self.wait_next_state()
                self.path = self.generate_path()
                self.start_time = time.time()
                self.best_index = 0
                velocity, stuck = self.follow_path(self.path)

            speed = velocity["speed"]
            ang_vel = velocity["ang_vel"]
            actions[idx] = [speed, ang_vel]

            print(f"speed and ang vel {speed:.3f} {ang_vel:.3f}")

            DEG_TO_RAD = math.pi / 180
            x, z, yaw = list(state[idx])

            print(f"current pos {x:.3f} {z:.3f} {yaw:.3f}")

            moderated_ang_vel = ang_vel * 0.13
            moderated_speed = speed * 0.9

            yaw_change = math.pi * (moderated_ang_vel + yaw / 180)

            new_x = x - moderated_speed * math.sin(yaw_change)
            new_z = z + moderated_speed * math.cos(yaw_change)

            new_yaw = (moderated_ang_vel * 180) + yaw
            print(f"maggie new pos {x:.3f} {z:.3f} {new_yaw:.3f}")
            print()

            
            self.agent_host.sendCommand(f"tp {new_x} 4.0 {new_z}")
            self.agent_host.sendCommand(f"setYaw {new_yaw}")
            time.sleep(0.1)

            self.expected_x = new_x
            self.expected_y = 4.0
            self.expected_z = new_z
            self.expected_yaw = new_yaw
            self.require_yaw_change = new_yaw != yaw
            self.require_move = new_x != x or new_z != z

            self.prev_state = self.wait_next_state()

        return (inputs, actions, state)

    def generate_dataset(self, path: Path, size=1000):
        current_size = 0
        with Progress() as progress:
            task = progress.add_task("Building dataset...", total=size)
            while current_size < size:
                batch = self.__next__()
                if batch is None:
                    continue
                inputs, actions, state = batch
                current_path = path / f"{time.time()}"
                os.makedirs(current_path, exist_ok=True)
                for t in range(len(inputs)):
                    image = ToPILImage()(inputs[t])
                    image.save(current_path / f"{t}.png")
                np.savez(current_path / "actions.npz", actions)
                np.savez(current_path / "state.npz", state)

                current_size += self.steps
                progress.update(task, advance=self.steps)

    # turn waiting from malmo python examples
    def wait_initial_state(self):
        """Before a command has been sent we wait for an observation of the world and a frame."""
        # wait for a valid observation
        world_state = self.agent_host.peekWorldState()
        while world_state.is_mission_running and all(
            e.text == "{}" for e in world_state.observations
        ):
            world_state = self.agent_host.peekWorldState()
        # wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while (
            world_state.is_mission_running
            and world_state.number_of_video_frames_since_last_state == num_frames_seen
        ):
            world_state = self.agent_host.peekWorldState()
        world_state = self.agent_host.getWorldState()

        if world_state.is_mission_running:

            assert len(world_state.video_frames) > 0, "No video frames!?"

            obs = json.loads(world_state.observations[-1].text)
            self.prev_x = obs["XPos"]
            self.prev_y = obs["YPos"]
            self.prev_z = obs["ZPos"]
            self.prev_yaw = obs["Yaw"]
            self.prev_dir = (
                self.prev_yaw
            )  # the direction of movement and yaw are detangled
            print(
                "Initial position:",
                self.prev_x,
                ",",
                self.prev_y,
                ",",
                self.prev_z,
                "yaw",
                self.prev_yaw,
            )

            self.prev_state = world_state
            self.init_pathfinding()

        return world_state

    def wait_next_state(self):
        """After each command has been sent we wait for the observation to change as expected and a frame."""
        # wait for the observation position to have changed
        print("Waiting for observation...", end=" ")
        obs = None
        while True:
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                print("mission ended.")
                break
            if not all(e.text == "{}" for e in world_state.observations):
                obs = json.loads(world_state.observations[-1].text)
                self.curr_x = obs["XPos"]
                self.curr_y = obs["YPos"]
                self.curr_z = obs["ZPos"]
                self.curr_yaw = math.fmod(obs["Yaw"], 360)
                if self.require_move:
                    if (
                        math.fabs(self.curr_x - self.prev_x) > self.tolerance
                        or math.fabs(self.curr_y - self.prev_y) > self.tolerance
                        or math.fabs(self.curr_z - self.prev_z) > self.tolerance
                    ):
                        break
                elif self.require_yaw_change:
                    if math.fabs(self.curr_yaw - self.prev_yaw) > self.tolerance:
                        break
                else:
                    break

        # wait for the render position to have changed
        print("Waiting for render...", end=" ")
        while True:
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                print("mission ended.")
                break
            if len(world_state.video_frames) > 0:
                # print('render changed')
                frame = world_state.video_frames[-1]
                curr_x_from_render = frame.xPos
                curr_y_from_render = frame.yPos
                curr_z_from_render = frame.zPos
                curr_yaw_from_render = math.fmod(frame.yaw, 360)
                if self.require_move:
                    # print('render move required')
                    if (
                        math.fabs(curr_x_from_render - self.prev_x) > self.tolerance
                        or math.fabs(curr_y_from_render - self.prev_y) > self.tolerance
                        or math.fabs(curr_z_from_render - self.prev_z) > self.tolerance
                    ):
                        #   print('render received a move.')

                        break
                elif self.require_yaw_change:
                    if math.fabs(curr_yaw_from_render - self.prev_yaw) > self.tolerance:
                        #  print('render received a turn.')
                        break
                else:
                    # print('render received.')
                    break

        num_frames_before_get = len(world_state.video_frames)
        world_state = self.agent_host.getWorldState()

        if world_state.is_mission_running:
            assert len(world_state.video_frames) > 0, "No video frames!?"
            num_frames_after_get = len(world_state.video_frames)
            assert (
                num_frames_after_get >= num_frames_before_get
            ), "Fewer frames after getWorldState!?"
            frame = world_state.video_frames[-1]
            # obs = json.loads( world_state.observations[-1].text )
            self.curr_x = obs["XPos"]
            self.curr_y = obs["YPos"]
            self.curr_z = obs["ZPos"]
            self.curr_yaw = math.fmod(
                obs["Yaw"], 360
            )  # math.fmod(180 + obs[u'Yaw'], 360)
            print(
                "1 New position from observation:",
                self.curr_x,
                ",",
                self.curr_y,
                ",",
                self.curr_z,
                "yaw",
                self.curr_yaw,
                end=" ",
            )
            if (
                math.fabs(self.curr_x - self.expected_x) > self.tolerance
                or math.fabs(self.curr_y - self.expected_y) > self.tolerance
                or math.fabs(self.curr_z - self.expected_z) > self.tolerance
                or math.fabs(self.curr_yaw - self.expected_yaw) > self.tolerance
            ):
                print(
                    " - ERROR DETECTED! Expected:",
                    self.expected_x,
                    ",",
                    self.expected_y,
                    ",",
                    self.expected_z,
                    "yaw",
                    self.expected_yaw,
                )
                # sys.exit("expected vs curr issue")
                return world_state
            else:
                pass
            #   print('as expected.')
            curr_x_from_render = frame.xPos
            curr_y_from_render = frame.yPos
            curr_z_from_render = frame.zPos
            # print('rendered yaw', frame.yaw)
            curr_yaw_from_render = math.fmod(
                frame.yaw, 360
            )  # math.fmod(180 + frame.yaw ,360)
            print(
                "New position from render:",
                curr_x_from_render,
                ",",
                curr_y_from_render,
                ",",
                curr_z_from_render,
                "yaw",
                curr_yaw_from_render,
            )
            if (
                math.fabs(curr_x_from_render - self.expected_x) > self.render_tolerance
                or math.fabs(curr_y_from_render - self.expected_y)
                > self.render_tolerance
                or math.fabs(curr_z_from_render - self.expected_z)
                > self.render_tolerance
                or math.fabs(curr_yaw_from_render - self.expected_yaw)
                > self.render_tolerance
            ):
                print(
                    " - ERROR DETECTED! Expected:",
                    self.expected_x,
                    ",",
                    self.expected_y,
                    ",",
                    self.expected_z,
                    "yaw",
                    self.expected_yaw,
                )
                # sys.exit("curr vs render issue")
                return world_state
            else:
                pass
            #   print('as expected.')
            self.prev_x = self.curr_x
            self.prev_y = self.curr_y
            self.prev_z = self.curr_z
            self.prev_yaw = self.curr_yaw

        return world_state
