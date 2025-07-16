import os

ROOT_DIR = os.path.dirname(__file__)
ENVS_DIR = os.path.join(ROOT_DIR,'Env')
#/home/pi/Downloads/humanoid-gym-main/humanoid/envs/__init__.py
#Taitan tinker_ppo Tinymal, Loong  humanoid_ppo
ROBOT_SEL = 'tinker_ppo'
#Trot Stand
GAIT_SEL = 'Trot'
PLAY_LAST = -1 #1採用最新的
PLAY_DIR ='/home/rx/Downloads/humanoid-gym-main/logs/Tinker_ppo/May08_09-15-29_/model_4000.pt'   
#Sim2Sim Cmd
SPD_X = 0.3
SPD_Y = 0.0
SPD_YAW = 0.0

#train param
MAX_ITER = 10000
SAVE_DIV = 1000


#./compile XX.urdf XX.xml
#rosrun robot_state_publisher robot_state_publisher my_robot.urdf