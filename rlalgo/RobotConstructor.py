from .RobotDDQN import RobotDDQN
from .RobotDQN import RobotDQN
from .RobotReinforce import RobotReinforce


def RobotConstructor(robot_config):
    if robot_config is None:
        return None
    if "type" in robot_config and robot_config["type"] == "provided":
        return robot_config["instance"].construct(robot_config)
    if robot_config["robot_type"] == "RobotReinforce":
        return RobotReinforce(robot_config)
    if robot_config["robot_type"] == "RobotDQN":
        return RobotDQN(robot_config)
    if robot_config["robot_type"] == "RobotDDQN":
        return RobotDDQN(robot_config)