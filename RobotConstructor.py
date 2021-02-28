from RobotReinforce import RobotReinforce


def RobotConstructor(robot_config):
    if robot_config is None:
        return None
    if robot_config["robot_type"] == "RobotReinforce":
        return RobotReinforce(robot_config)