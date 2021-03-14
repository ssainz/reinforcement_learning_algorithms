from mysql.connector import (connection)

from .Robot import Robot
from .Env import Env
from .World import World

# DB:
db_create_script = '''
 create table rlresult(
     group_id int, 
     result_id int,
 robot_config_idint,
 env_config_id int,
     step int,
     reward DOUBLE,
     reward1 DOUBLE,
     reward2 DOUBLE,
     reward3 DOUBLE,
     reward4 DOUBLE
 );
 drop table robot_config;
 create table robot_config(
 robot_config_id int,
 robot_type VARCHAR(100),
 net_config_layers VARCHAR(1000),
 net_config_nonlinear VARCHAR(100),
 gamma DOUBLE,
 lr DOUBLE,
 robot_config JSON
 );
 drop table env_config;
 create table env_config(
 env_config_id int,
 env_type VARCHAR(50),
 name VARCHAR(200),
 env_config JSON
 );
'''


class dao:
    def __init__(self):
        self.conn = connection.MySQLConnection(user='ssp',
                                               password='ssp',
                                               host='192.168.0.6',
                                               database='rlresults')
    def save_world(self, world: World):
        id = self.get_latest_result_id() + 1
        robot_id = self.save_robot(world.robot)
        env_id = self.save_env(world.env)
        step = 1
        for result in world.results:
            if len(result) == 1:
                self.run_insert("insert into rlresult (group_id, result_id, robot_config_id, env_config_id, step, reward) values (%s, %s, %s, %s, %s, %s)" % (
                    str(world.group_id),
                    str(id),
                    str(robot_id),
                    str(env_id),
                    str(step),
                    str(result[0])
                ))
            elif len(result) == 4:
                self.run_insert("insert into rlresult (group_id, result_id, robot_config_id, env_config_id, step, reward, reward1, reward2, reward3) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)" % (
                    str(world.group_id),
                    str(robot_id),
                    str(env_id),
                    str(step),
                    str(result[0]),
                    str(result[1]),
                    str(result[2]),
                    str(result[3])
                ))
            step += 1
    def save_robot(self, robot: Robot):
        id = self.get_latest_robot_id() + 1
        self.run_insert("insert into robot_config ("
                        "robot_config_id,"
                        "robot_type,"
                        "net_config_layers,"
                        "net_config_nonlinear,"
                        "gamma,"
                        "lr,"
                        "robot_config"
                        ") values (%s, '%s', '%s', '%s', %s, %s, %s)" % (
            str(id),
            str(robot.config["robot_type"]),
            str(robot.config["net_config"]["layers"]),
            str(robot.config["net_config"]["non_linear_function"]),
            str(robot.config["gamma"]),
            str(robot.config["lr"]),
            "'"+to_json(str(robot.config))+"'"
        ))
        return id
    def get_latest_robot_id(self):
        query = "select max(robot_config_id) from robot_config"
        results = self.run_query(query)
        id = 0
        for tuple in results:
            id = tuple[0] if tuple[0] is not None else 0
        return id
    def save_env(self, env: Env):
        id = self.get_latest_env_id() + 1
        self.run_insert("insert into env_config ("
                        "env_config_id,"
                        "env_type,"
                        "name,"
                        "env_config"
                        ") values (%s, '%s', '%s','%s')" % (
            str(id),
            env.env_config["env_type"],
            env.env_config["name"],
            to_json(str(env.env_config))
        ))
        return id
    def get_latest_env_id(self):
        query = "select max(env_config_id) from env_config"
        results = self.run_query(query)
        id = 0
        for tuple in results:
            id = tuple[0] if tuple[0] is not None else 0
        return id
    def get_latest_result_id(self):
        query = "select max(result_id) from rlresult"
        results = self.run_query(query)
        id = 0
        for tuple in results:
            id = tuple[0] if tuple[0] is not None else 0
        return id
    def get_latest_group_id(self):
        query = "select max(group_id) from rlresult"
        results = self.run_query(query)
        id = 0
        for tuple in results:
            id = tuple[0] if tuple[0] is not None else 0
        return id
    def run_query(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        results = []
        for tuple in cursor:
            results.append(tuple)
        cursor.close()
        return results
    def run_insert(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        self.conn.commit()
    def close(self):
        self.conn.close()
def to_json(json_str):
    a = json_str
    b = a.replace("'", '"')
    c = b.replace("False","false")
    d = c.replace("True","true")
    return d