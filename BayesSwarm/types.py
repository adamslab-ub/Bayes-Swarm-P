class SimulationConfigs:
    def __init__(self,
                 simulation_mode="light",  # "pybullet"
                 environment="None", # "plain"
                 texture=None,  # "source"
                 robot_type=None): # "uav", "ugv"
        self.mode = simulation_mode
        self.environment = environment
        self.texture = texture
        self.robot_type = robot_type
