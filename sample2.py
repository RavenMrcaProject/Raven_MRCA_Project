# set a dynamic lib to path
import orca_module
import logging
import math
import os
from typing import Any, List
import numpy as np
import plotly.graph_objects as go
from staliro.core.interval import Interval
from staliro.core import best_eval, best_run
from staliro.core.model import FailureResult
from staliro.core.result import worst_eval, worst_run
from staliro.models import State, ode, blackbox, BasicResult, Model, ModelInputs, ModelResult, Trace
from staliro.optimizers import UniformRandom, DualAnnealing
from staliro.options import Options
from staliro.specifications import RTAMTDense, RTAMTDiscrete
from staliro.staliro import simulate_model, staliro
import math
from decimal import Decimal, ROUND_HALF_UP, localcontext
import random


def parse_config(filename):
    config = {}
    with open(filename, 'r') as file:
        key, value = None, ""
        for line in file:
            # Remove whitespace and skip comments
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # Check if line continues a previous value
            if line.startswith('"') and key is not None:
                value += line.strip('"')
                config[key] = value
                key, value = None, ""
                continue

            # Split on the first equals sign
            parts = line.split('=', 1)
            if len(parts) != 2:
                print(f"Skipping invalid line: {line}")
                continue

            key, value = parts[0].strip(), parts[1].strip().strip('"')

            # Check for multi-line values
            if value.endswith("\\"):
                value = value[:-1].strip('"')
            else:
                config[key] = value
                key, value = None, ""

    return config


algorithm = "orca"
# TODO: change path of config.txt
config = parse_config("config.txt")
num_of_robots = int(config.get("num_robots", 0))
radius = float(config.get("radius", 0.0))
time_step = float(config.get("time_step", 0.0))
environment_file = config.get("environment_file", "")
environment_name = os.path.basename(os.path.normpath(environment_file))
maxSpeed = float(config.get("maxSpeed", 0.0))
minPosChange = float(config.get("minPosChange", 0.0))
maxPosChange = float(config.get("maxPosChange", 0.0))
minVelChange = float(config.get("minVelChange", 0.0))
maxVelChange = float(config.get("maxVelChange", 0.0))
numberOfFalseMessage = int(config.get("numberOfFalseMessage", 0))
attackType = int(config.get("attackType", 0))
victimRobotId = int(config.get("victimRobotId", 0))
attackedRobotId = int(config.get("attackedRobotId", 0))
deadlockTimestep = int(config.get("deadlockTimestep", 0))
deadlockPosChange = float(config.get("deadlockPosChange", 0.0))
totalTimeStep = float(config.get("totalTimeStep", 0.0))
falsificationIterations = int(config.get("falsificationIterations", 0))
falsificationRuns = int(config.get("falsificationRuns", 0))
delayConstant = int(config.get("delayConstant", 0))
pointX = float(config.get("pointX", 0))
pointY = float(config.get("pointY", 0))


def print_all_config_params():
    print("num_of_robots: ", num_of_robots)
    print("radius: ", radius)
    print("time_step: ", time_step)
    print("environment_file: ", environment_file)
    print("environment_name: ", environment_name)
    print("maxSpeed: ", maxSpeed)
    print("minPosChange: ", minPosChange)
    print("maxPosChange: ", maxPosChange)
    print("minVelChange: ", minVelChange)
    print("maxVelChange: ", maxVelChange)
    print("numberOfFalseMessage: ", numberOfFalseMessage)
    print("attackType: ", attackType)
    print("victimRobotId: ", victimRobotId)
    print("attackedRobotId: ", attackedRobotId)
    print("deadlockTimestep: ", deadlockTimestep)
    print("deadlockPosChange: ", deadlockPosChange)
    print("totalTimeStep: ", totalTimeStep)
    print("falsificationIterations: ", falsificationIterations)
    print("falsificationRuns: ", falsificationRuns)
    print("delayConstant: ", delayConstant)
    print("pointX: ", pointX)
    print("pointY: ", pointY)


# change 5 if you want to change the number of parameters staliro wants to predict
def process_initial_conditions(initial_conditions):
    print("Initial conditions: ", initial_conditions)
    result = {}
    for i in range(0, len(initial_conditions), 3):
        key = round_to_nearest_step(initial_conditions[i], step=time_step)
        values = tuple(initial_conditions[i+1:i+3])
        result[key] = values
    return result


def round_to_nearest_step(number, step=0.25):
    number = Decimal(str(number))
    step = Decimal(str(step))
    rounded_value = (number / step).quantize(Decimal('1'),
                                             rounding=ROUND_HALF_UP) * step
    return float(rounded_value)


def increment_precise(number, increment, precision=10):
    with localcontext() as ctx:
        ctx.prec = precision
        num_decimal = Decimal(str(number))
        inc_decimal = Decimal(str(increment))
        result = num_decimal + inc_decimal
        result = result.quantize(Decimal('0.0'), rounding=ROUND_HALF_UP)
    return float(result)


def calculate_abs_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def parse_obstacles(file_path):
    obstacles = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # every obstacle consists of 4 vertices
        for i in range(0, len(lines), 4):
            obstacle = [tuple(map(float, line.split(', ')))
                        for line in lines[i:i+4]]
            obstacles.append(obstacle)
    return obstacles


def parse_obstacles_onebyone(file_path):
    obstacles = []
    with open(file_path, 'r') as file:
        for line in file:
            obstacle = tuple(map(float, line.strip().split(", ")))
            obstacles.append(obstacle)
    return obstacles


def parse_goals(file_path):
    goals = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line by comma and convert each part to a float
            x, y = map(float, line.strip().split(', '))
            goals.append((x, y))
    return goals


obstacles = parse_obstacles_onebyone(
    environment_file + "obstacles.txt")
print(environment_file)
print("Obstacles: ", obstacles)
goal_positions = parse_goals(environment_file + "goals.txt")
print("Goal positions: ", goal_positions)


def navigation_distance(file_path):
    def parse_coordinates(filename):
        with open(filename, 'r') as file:
            line = file.readline().strip()
            x, y = map(float, line.split(', '))
            return x, y

    def calculate_euclidean_distance(point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    destination = parse_coordinates(file_path + 'goals.txt')
    robot_initial_position = parse_coordinates(file_path + 'robot_pos.txt')

    return calculate_euclidean_distance(destination, robot_initial_position)


def point_line_distance(point, line_start, line_end):
    # Convert points to numpy arrays for vector operations
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    # Calculate the vector from line_start to point and line_start to line_end
    line_vector = line_end - line_start
    point_vector = point - line_start

    # Calculate the projection of the point_vector onto the line_vector
    line_length_squared = np.dot(line_vector, line_vector)
    if line_length_squared == 0:
        # line_start and line_end are the same point
        return np.linalg.norm(point_vector)

    projection = np.dot(point_vector, line_vector) / line_length_squared

    if projection < 0:
        # The projection falls before line_start
        closest_point = line_start
    elif projection > 1:
        # The projection falls after line_end
        closest_point = line_end
    else:
        # The projection falls between line_start and line_end
        closest_point = line_start + projection * line_vector

    # Calculate the distance from the point to the closest point on the line segment
    return np.linalg.norm(point - closest_point)


# create log of robot positions for last falsification
global_log_list_last = []
global_log_list_prev = []
global_log_list_fake = []
global_log_list_last_with_velocities = []

# create log of robot positions
global_log_list = []

global_min_dist_of_victim_to_static_so_far = float('inf')
global_min_dist_of_any_to_static_so_far = float('inf')
global_min_dist_of_victim_to_robots_so_far = float('inf')
global_min_dist_of_any_to_robots_so_far = float('inf')
global_min_navigating_to_point_victim = float('inf')
global_min_navigating_to_point_any = float('inf')
global_min_deadlock_of_victim_robot = float('inf')
global_min_deadlock_of_any_robot = float('inf')
global_min_navigation_duration_of_victim_robot = float('inf')
global_max_navigation_duration_of_any_robot = -float('inf')

global_iteration = 0


@blackbox()
def Orca_Model(state: State, time: Interval, _: Any) -> BasicResult:
    global global_iteration
    global_iteration += 1

    global global_log_list_last
    global global_log_list_prev
    global global_log_list_fake
    global global_log_list_last_with_velocities

    global global_log_list
    global global_min_dist_of_victim_to_static_so_far
    global global_min_dist_of_any_to_static_so_far
    global global_min_dist_of_victim_to_robots_so_far
    global global_min_dist_of_any_to_robots_so_far
    global global_min_navigating_to_point_victim
    global global_min_navigating_to_point_any
    global global_min_deadlock_of_victim_robot
    global global_min_deadlock_of_any_robot
    global global_min_navigation_duration_of_victim_robot
    global global_max_navigation_duration_of_any_robot

    counter = 0.0

    time_result = []
    list_min_obstacle_dist_of_victim = []
    list_min_obstacle_dist_of_any_robot = []
    list_min_inter_robot_dist_of_victim_robot = []
    list_min_inter_robot_dist_of_any_robot = []
    list_min_deadlock_of_victim_robot = []
    list_min_deadlock_of_any_robot = []
    list_min_navigation_duration_of_victim_robot = []
    list_max_navigation_duration_of_any_robot = []
    list_min_navigating_to_point_victim = []
    list_min_navigating_to_point_any = []
    list_min_deadlock_for_navigation_duration_of_any_robot = []

    last_positions = []
    last_positions_dict = {robot_id: [] for robot_id in range(num_of_robots)}

    orca_instance = orca_module.init_orca()

    mapped_conditions = process_initial_conditions(state)
    print("Mapped conditions: ")
    print(mapped_conditions)
    keys_list_times = list(mapped_conditions.keys())

    local_dist_victim = float('inf')
    local_pos_list_for_dist_victim = []
    local_dist_any = float('inf')
    local_pos_list_for_dist_any = []
    local_min_obstacle_dist_of_victim = float('inf')
    local_pos_list_for_obstacle_dist_of_victim = []
    local_min_obstacle_dist_of_any_robot = float('inf')
    local_pos_list_for_obstacle_dist_of_any_robot = []
    local_min_inter_robot_dist_of_victim_robot = float('inf')
    local_pos_list_for_inter_robot_dist_of_victim_robot = []
    local_min_inter_robot_dist_of_any_robot = float('inf')
    local_pos_list_for_inter_robot_dist_of_any_robot = []
    local_min_deadlock_of_victim_robot = float('inf')
    local_pos_list_for_deadlock_of_victim_robot = []
    local_min_deadlock_of_any_robot = float('inf')
    local_pos_list_for_deadlock_of_any_robot = []
    local_min_navigation_duration_of_victim_robot = float('inf')
    local_pos_list_for_navigation_duration_of_victim_robot = []
    local_max_navigation_duration_of_any_robot = -float('inf')
    local_pos_list_for_navigation_duration_of_any_robot = []

    delay_constant = 1
    # check if the attack is navigation delay attack
    if attackType == 9 or attackType == 10:
        delay_constant = delayConstant

    try:
        temp_list_for_global = []
        temp_list_for_global_prev = []
        temp_list_for_global_fake = []
        temp_list_for_global_with_velocities = []

        # run orca simulation with timestep
        while counter < (totalTimeStep * delay_constant):
            time_result.append(counter)
            # print("=======")
            # print("Counter: ", counter)

            updated_positions = orca_module.get_agent_positions(orca_instance)
            updated_velocities = orca_module.get_agent_positions(orca_instance)

            prev_pos = updated_positions[attackedRobotId]
            # print("Prev pos: ", prev_pos)

            # print real logs
            # temp_list_for_global_prev.append(counter)
            # for updated_position in updated_positions:
            #     temp_list_for_global_prev.append(updated_position)

            # plan attack
            if counter in keys_list_times:
                value = mapped_conditions[counter]
                # print("Value: ", value)
                updated_positions[attackedRobotId] = (
                    updated_positions[attackedRobotId][0] + value[0], updated_positions[attackedRobotId][1] + value[1], updated_positions[attackedRobotId][2])
                # print("Updated positions: ", updated_positions)
                orca_module.set_agent_positions(orca_instance,
                                                updated_positions)
            # if counter == 1.0:
            #     print("hey1")
            #     updated_positions[attackedRobotId] = (
            #         updated_positions[attackedRobotId][0] - -2.5013652271882165, updated_positions[attackedRobotId][1] - 0.220594090714485, updated_positions[attackedRobotId][2])
            #     # print("Updated positions: ", updated_positions)
            #     orca_module.set_agent_positions(orca_instance,
            #                                     updated_positions)
            # if counter == 16:
            #     print("hey2")
            #     updated_positions[attackedRobotId] = (
            #         updated_positions[attackedRobotId][0] - 1.5127996252565765, updated_positions[attackedRobotId][1] + 1.9568271970817364, updated_positions[attackedRobotId][2])
            #     # print("Updated positions: ", updated_positions)
            #     orca_module.set_agent_positions(orca_instance,
            #                                     updated_positions)
            # if counter == 0:
            #     print("hey3")
            #     updated_positions[attackedRobotId] = (
            #         updated_positions[attackedRobotId][0] - 1.0911954874609364, updated_positions[attackedRobotId][1] + 0.020084566334844567, updated_positions[attackedRobotId][2])
            #     # print("Updated positions: ", updated_positions)
            #     orca_module.set_agent_positions(orca_instance,
            #                                     updated_positions)

            # print fake logs
            # temp_list_for_global_fake.append(counter)
            # for updated_position in updated_positions:
            #     temp_list_for_global_fake.append(updated_position)

            orca_module.calculate_next_positions_(
                orca_instance, attackedRobotId, prev_pos)
            updated_positions = orca_module.get_agent_positions(orca_instance)
            # print("Updated positions: ", updated_positions)

            # ==========================================
            # Attack measurements
            # ==========================================

            # ==========================================
            # decide collision of victim robot with obstacles:

            min_distance_to_obstacle = float('inf')
            victim_position = updated_positions[victimRobotId]
            for obstacle_id, obstacle_pos in enumerate(obstacles):
                # Calculate the Euclidean distance
                distance = ((obstacle_pos[0] - victim_position[0]) ** 2 +
                            (obstacle_pos[1] - victim_position[1]) ** 2) ** 0.5
                # check all robots
                min_distance_to_obstacle = min(
                    min_distance_to_obstacle, distance)

            # save for falsification
            list_min_obstacle_dist_of_victim.append(min_distance_to_obstacle)

            local_min_obstacle_dist_of_victim = min(
                local_min_obstacle_dist_of_victim, min_distance_to_obstacle)
            local_pos_list_for_obstacle_dist_of_victim.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_obstacle_dist_of_victim.append(
                    updated_position)

            # ==========================================
            # decide collision of victim robot with other robots:

            min_dist_of_inter_robot_to_victim_robot = float('inf')
            victim_position = updated_positions[victimRobotId]
            for robot_id, robot_pos in enumerate(updated_positions):
                if robot_id == victimRobotId or robot_id == attackedRobotId:
                    continue  # Skip the victim robot

                # Calculate the Euclidean distance
                min_inter_robot_dist = ((robot_pos[0] - victim_position[0]) ** 2 +
                                        (robot_pos[1] - victim_position[1]) ** 2) ** 0.5
                # check all robots
                min_dist_of_inter_robot_to_victim_robot = min(
                    min_dist_of_inter_robot_to_victim_robot, min_inter_robot_dist)

                # print for me
                # if min_dist_of_inter_robot_to_victim_robot < radius * 2:
                # print("Collision with other robots at time:", counter, " between robots",
                #   victimRobotId, "and", robot_id, "with positions", victim_position, robot_pos)
            # save for falsification
            list_min_inter_robot_dist_of_victim_robot.append(
                min_dist_of_inter_robot_to_victim_robot)

            local_min_inter_robot_dist_of_victim_robot = min(
                local_min_inter_robot_dist_of_victim_robot, min_dist_of_inter_robot_to_victim_robot)
            local_pos_list_for_inter_robot_dist_of_victim_robot.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_inter_robot_dist_of_victim_robot.append(
                    updated_position)

            # ==========================================
            # decide herding attack for victim robot
            # dist: difference of static point and victim's position
            dist_victim_point = calculate_abs_distance(pointX, pointY,
                                                       updated_positions[victimRobotId][0], updated_positions[victimRobotId][1])
            # save for falsification
            list_min_navigating_to_point_victim.append(dist_victim_point)

            # save to print
            local_dist_victim = min(local_dist_victim, dist_victim_point)
            local_pos_list_for_dist_victim.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_dist_victim.append(updated_position)

            # ==========================================
            # decide deadlock of victim robot:

            last_positions.append(updated_positions[victimRobotId])
            # Keep only the last positions
            if len(last_positions) > deadlockTimestep:
                last_positions.pop(0)

            total_pos_change = 0.0
            if len(last_positions) == deadlockTimestep:
                # sum of the move changes between consecutive positions in last_positions
                for i in range(len(last_positions) - 1):
                    total_pos_change += calculate_abs_distance(
                        last_positions[i][0], last_positions[i][1], last_positions[i+1][0], last_positions[i+1][1])
            else:
                total_pos_change = float('inf')

            # edge case
            # skip last moments if the robot is close to the goal within 10 meters and does not move more.
            if calculate_abs_distance(updated_positions[victimRobotId][0], updated_positions[victimRobotId][1], goal_positions[victimRobotId][0], goal_positions[victimRobotId][1]) < 5.0:
                total_pos_change = float('inf')

            # print for me
            # if total_pos_change < deadlockPosChange:
                # print("Deadlock of victim robot at time:", counter, " with position",
                #   updated_positions[victimRobotId], " and measurement: ", total_pos_change)

            # save for falsification
            list_min_deadlock_of_victim_robot.append(
                total_pos_change)

            # save to print
            local_min_deadlock_of_victim_robot = min(
                local_min_deadlock_of_victim_robot, total_pos_change)
            local_pos_list_for_deadlock_of_victim_robot.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_deadlock_of_victim_robot.append(
                    updated_position)

            # ==========================================
            # decide navigation delay of victim robot:

            current_position = updated_positions[victimRobotId]
            remaining_distance = calculate_abs_distance(
                current_position[0], current_position[1], goal_positions[victimRobotId][0], goal_positions[victimRobotId][1])

            # save for falsification
            list_min_navigation_duration_of_victim_robot.append(
                remaining_distance)

            # save to print
            local_min_navigation_duration_of_victim_robot = min(
                local_min_navigation_duration_of_victim_robot, remaining_distance)
            local_pos_list_for_navigation_duration_of_victim_robot.append(
                counter)
            for updated_position in updated_positions:
                local_pos_list_for_navigation_duration_of_victim_robot.append(
                    updated_position)

            # ==========================================
            temp_list_for_global.append(counter)
            temp_list_for_global_with_velocities.append(counter)
            for updated_position in updated_positions:
                temp_list_for_global.append(updated_position)
                temp_list_for_global_with_velocities.append(updated_position)
            # for updated_velocity in updated_velocities:
                # temp_list_for_global_with_velocities.append(updated_velocity)

            # counter += time_step
            counter = increment_precise(counter, time_step, 10)

        if attackType == 1:
            # victim - static obstacle collision
            if local_min_obstacle_dist_of_victim < global_min_dist_of_victim_to_static_so_far:
                global_min_dist_of_victim_to_static_so_far = local_min_obstacle_dist_of_victim
                global_log_list = local_pos_list_for_obstacle_dist_of_victim
        elif attackType == 3:
            # victim robot - other robots collision
            if local_min_inter_robot_dist_of_victim_robot < global_min_dist_of_victim_to_robots_so_far:
                global_min_dist_of_victim_to_robots_so_far = local_min_inter_robot_dist_of_victim_robot
                global_log_list = local_pos_list_for_inter_robot_dist_of_victim_robot
        elif attackType == 5:
            # navigating the point - victim robot
            if local_dist_victim < global_min_navigating_to_point_victim:
                global_min_navigating_to_point_victim = local_dist_victim
                global_log_list = local_pos_list_for_dist_victim
        elif attackType == 7:
            # victim robot - deadlock
            if local_min_deadlock_of_victim_robot < global_min_deadlock_of_victim_robot:
                global_min_deadlock_of_victim_robot = local_min_deadlock_of_victim_robot
                global_log_list = local_pos_list_for_deadlock_of_victim_robot
        elif attackType == 9:
            # victim robot - navigation delay
            if local_min_navigation_duration_of_victim_robot < global_min_navigation_duration_of_victim_robot:
                global_min_navigation_duration_of_victim_robot = local_min_navigation_duration_of_victim_robot
                global_log_list = local_pos_list_for_navigation_duration_of_victim_robot

    except Exception as e:
        print("Errors related to falsification!")
        print(e)
    try:
        print("global_iteration: ", global_iteration)
        print("time_result:", time_result, len(time_result))
        print("list_min_obstacle_dist_of_victim:",
              list_min_obstacle_dist_of_victim, len(
                  list_min_obstacle_dist_of_victim)
              )
        print("list_min_obstacle_dist_of_any_robot:",
              list_min_obstacle_dist_of_any_robot, len(list_min_obstacle_dist_of_any_robot))
        print("list_min_inter_robot_dist_of_victim_robot:",
              list_min_inter_robot_dist_of_victim_robot, len(list_min_inter_robot_dist_of_victim_robot))
        print("list_min_inter_robot_dist_of_any_robot:",
              list_min_inter_robot_dist_of_any_robot, len(
                  list_min_inter_robot_dist_of_any_robot))
        print("list_min_navigating_to_point_victim:",
              list_min_navigating_to_point_victim, len(list_min_navigating_to_point_victim))
        print("list_min_navigating_to_point_any:",
              list_min_navigating_to_point_any, len(list_min_navigating_to_point_any))
        print("list_min_deadlock_of_victim_robot:",
              list_min_deadlock_of_victim_robot, len(list_min_deadlock_of_victim_robot))
        print("list_min_deadlock_of_any_robot:",
              list_min_deadlock_of_any_robot, len(list_min_deadlock_of_any_robot))
        print("list_min_navigation_duration_of_victim_robot:",
              list_min_navigation_duration_of_victim_robot, len(list_min_navigation_duration_of_victim_robot))
        print("list_max_navigation_duration_of_any_robot:",
              list_max_navigation_duration_of_any_robot, len(list_max_navigation_duration_of_any_robot))
        print("list_min_deadlock_for_navigation_duration_of_any_robot:",
              list_min_deadlock_for_navigation_duration_of_any_robot, len(list_min_deadlock_for_navigation_duration_of_any_robot))

        trace = Trace(time_result, [time_result,
                                    list_min_obstacle_dist_of_victim,
                                    [0.0] * len(time_result),
                                    list_min_inter_robot_dist_of_victim_robot,
                                    [0.0] * len(time_result),
                                    list_min_navigating_to_point_victim,
                                    [0.0] * len(time_result),
                                    list_min_deadlock_of_victim_robot,
                                    [0.0] * len(time_result),
                                    list_min_navigation_duration_of_victim_robot,
                                    [0.0] * len(time_result),
                                    [0.0] * len(time_result)])
        # trace = Trace(time_result, [time_result,
        #                             list_min_obstacle_dist_of_victim,
        #                             [0.0] * len(time_result),
        #                             list_min_inter_robot_dist_of_victim_robot,
        #                             [0.0] * len(time_result),
        #                             list_min_navigating_to_point_victim,
        #                             [0.0] * len(time_result),
        #                             list_min_deadlock_of_victim_robot,
        #                             [0.0] * len(time_result),
        #                             list_min_navigation_duration_of_victim_robot,
        #                             [0.0] * len(time_result),
        #                             [0.0] * len(time_result)])
        global_log_list_last = temp_list_for_global
        global_log_list_prev = temp_list_for_global_prev
        global_log_list_fake = temp_list_for_global_fake
        global_log_list_last_with_velocities = temp_list_for_global_with_velocities
        # print("global_log_list_last: ", global_log_list_last)
        # print("result:")
        # print(trace)
        return BasicResult(trace)
    except Exception as e:
        print("Exception occurred during simulation")
        return FailureResult()


state_matcher = {
    "t": 0,
    "min_dist_of_victim_to_obstacles": 1,
    "min_dist_of_any_robot_to_obstacles": 2,
    "min_of_inter_robot_dis_of_victim_robot": 3,
    "min_of_inter_robot_dis_of_any_robot": 4,
    "min_dist_victim": 5,
    "min_dist_any": 6,
    "deltaPos_VictimRobot": 7,
    "deltaPos_AnyRobot": 8,
    "distToGoal_VictimRobot": 9,
    "distToGoal_AnyRobot": 10,
    "deltaPos_Any_for_NavigationDelay": 11
}

# Decide attack type
specification = None
attackName = ""
if attackType == 1:
    attackName = "collision_obstacle_victim_robot"
    phi_collision_obstacle_victim_drone = f"always (min_dist_of_victim_to_obstacles > {radius + math.sqrt(3)})"
    print("phi_collision_obstacle_targeted_drone: ",
          phi_collision_obstacle_victim_drone)
    specification = RTAMTDense(
        phi_collision_obstacle_victim_drone, state_matcher)
elif attackType == 3:
    attackName = "collision_btw_victim_and_other_robots"
    phi_collision_btw_victim_and_other_robots = f"always (min_of_inter_robot_dis_of_victim_robot > {radius*2})"
    print("phi_collision_btw_victim_and_other_robots: ",
          phi_collision_btw_victim_and_other_robots)
    specification = RTAMTDense(
        phi_collision_btw_victim_and_other_robots, state_matcher)
elif attackType == 5:
    attackName = "navigating_to_point_targeted"
    phi_navigating_robot_targeted = "always (min_dist_victim > 0.5)"
    print("phi_navigating_victim: ", phi_navigating_robot_targeted)
    specification = RTAMTDense(
        phi_navigating_robot_targeted, state_matcher)
elif attackType == 7:
    attackName = "deadlock_of_victim_robot"
    phi_deadlock_of_victim_robot = f"always deltaPos_VictimRobot > {deadlockPosChange}"
    print("phi_deadlock_of_victim_robot: ", phi_deadlock_of_victim_robot)
    specification = RTAMTDense(
        phi_deadlock_of_victim_robot, state_matcher)
elif attackType == 9:
    attackName = "navigation_delay_of_victim_robot"
    phi_navigation_delay_of_victim_robot = f"eventually (distToGoal_VictimRobot <= 1.0)"
    # phi_navigation_delay_of_victim_robot = f"eventually (distToGoal_VictimRobot <= 1.0)"
    print("phi_navigation_delay_of_victim_robot: ",
          phi_navigation_delay_of_victim_robot)
    specification = RTAMTDense(
        phi_navigation_delay_of_victim_robot, state_matcher)
else:
    print("Invalid attack type!")
    exit()


print_all_config_params()

# message spoofing variables
initial_conditions = [
    (0, totalTimeStep),
    (-5, 5),
    (-5, 5)
] * numberOfFalseMessage

optimizer = DualAnnealing()
flag = 1
i = -1
while i < falsificationRuns and flag:
    i += 1
    options = Options(runs=1, iterations=falsificationIterations, interval=(
        0, 10), static_parameters=initial_conditions, seed=random.randint(0, 2**32 - 1))
    result = staliro(Orca_Model, specification, optimizer, options)

    worst_run_ = worst_run(result)
    worst_sample = worst_eval(worst_run_).sample
    worst_result = simulate_model(Orca_Model, options, worst_sample)

    print("\nWorst Sample:")
    print(worst_sample)

    print("\nResult:")
    print(worst_result.trace.states)

    print("\nWorst Evaluation:")
    print(worst_eval(worst_run(result)))

    if worst_eval(worst_run_).cost < 0:
        flag = 0

        # if evaluation cost is negative, then print
    if worst_eval(worst_run_).cost < 0:
        print("Found the attack!")
        print("Evaluation cost: ", worst_eval(worst_run_).cost)
        with open(f'/Attack/src/falsiParams/{attackName}_ite_{numberOfFalseMessage}mes_{environment_name}_{algorithm}_{radius}_{time_step}.txt', 'w') as file:
            file.write(str(worst_eval(worst_run(result))))
        with open(f'/Attack/src/falsiResults3/{attackName}_run_positions_{falsificationIterations}ite_{numberOfFalseMessage}mes_{environment_name}_{algorithm}_{radius}_{time_step}.txt', 'w') as file:
            for i in range(0, len(global_log_list_last), num_of_robots + 1):
                time = global_log_list_last[i]
                positions = global_log_list_last[i+1:i+num_of_robots+1]
                positions_str = ' '.join(
                    [f"({pos[0]:.4f},{pos[1]:.4f})" for pos in positions])
                file.write(f"{time} {positions_str}\n")
        print("End2")
    else:
        print("Could not find an attack!")
