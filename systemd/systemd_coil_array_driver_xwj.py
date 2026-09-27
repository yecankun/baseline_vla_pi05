# import sys
# sys.path.append("/home/dengxutian/GitHub/2025_haptic_digital_twin-main")
#
# import json
# import socket
# from utils.utils_hardware import Robot
#
# LOCAL_IP = "192.168.137.3"
# LOCAL_PORT = 12301
#
# server_socket = socket.socket()
# server_socket.bind((LOCAL_IP, LOCAL_PORT)) # 设置端口
# server_socket.listen(5) # 设置排队数
# print(f"Driver is listening at {LOCAL_IP}:{LOCAL_PORT}")
#
# robot = Robot()
# client_socket, client_address = server_socket.accept()
# print(f"Driver is connected to {client_address[0]}:{client_address[1]}")
#
# while True:
#
#     degree = client_socket.recv(1024)
#     degree = degree.decode("utf-8")
#     print(f"Robot is moving to degree: {degree}")
#     robot.to(float(degree))
#
#     robot_dof = robot.dof
#     robot_deg = robot.degree
#     message = {
#         "robot_dof": robot_dof,
#         "robot_deg": robot_deg,
#     }
#     message = json.dumps(message).encode("utf-8")
#     client_socket.send(message)
#
#     import sys
#
#     sys.path.append("/home/dengxutian/GitHub/2025_haptic_digital_twin-main")
#
#     import json
#     import socket
#     from utils.utils_hardware import Robot
#
#     LOCAL_IP = "192.168.137.3"
#     LOCAL_PORT = 12301
#
#     server_socket = socket.socket()
#     server_socket.bind((LOCAL_IP, LOCAL_PORT))  # 设置端口
#     server_socket.listen(5)  # 设置排队数
#     print(f"Driver is listening at {LOCAL_IP}:{LOCAL_PORT}")
#
#     robot = Robot()
#     client_socket, client_address = server_socket.accept()
#     print(f"Driver is connected to {client_address[0]}:{client_address[1]}")
#
#     while True:
#         degree = client_socket.recv(1024)
#         degree = degree.decode("utf-8")
#         print(f"Robot is moving to degree: {degree}")
#         robot.to(float(degree))
#
#         robot_dof = robot.dof
#         robot_deg = robot.degree
#         message = {
#             "robot_dof": robot_dof,
#             "robot_deg": robot_deg,
#         }
#         message = json.dumps(message).encode("utf-8")
#         client_socket.send(message)

import sys
import os
import time

sys.path.append("/home/dengxutian/GitHub/2025_haptic_digital_twin-main")

import json
import socket
from utils.utils_hardware import Robot

LOCAL_IP = "192.168.137.3"
LOCAL_PORT = 12301

server_socket = socket.socket()
server_socket.bind((LOCAL_IP, LOCAL_PORT))  # 设置端口
server_socket.listen(5)  # 设置排队数
print(f"Driver is listening at {LOCAL_IP}:{LOCAL_PORT}")

robot = Robot()
client_socket, client_address = server_socket.accept()
print(f"Driver is connected to {client_address[0]}:{client_address[1]}")

current_dir = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(current_dir, "time.txt")
if not os.path.exists(file):
    print(f'文件{file}不存在，创建文件')
    with open(file, 'w') as f:
        pass

def save_time(*args):
    args_string = ' '.join(map(str, args))
    write_in(file, args_string)

def write_in(file_name: str, data):
    with open(file_name, encoding="utf-8", mode="a") as file:
        file.write(str(data) + '\n')


while True:

    raw = client_socket.recv(1024)
    receive_time = time.time()
    if not raw:
        break

    try:
        msg = json.loads(raw.decode("utf-8"))
        pa = float(msg["percent_a"])
        pb = float(msg["percent_b"])
        pc = float(msg["percent_c"])

    except (ValueError, KeyError) as e:
        print("接收到的消息格式错误：", e, raw)
        continue

    print(f"设置线圈 PWM 百分比：A={pa}, B={pb}, C={pc}")
    robot.to(pa, pb, pc)  # 三通道
    save_time(receive_time,time.time(),pa,pb,pc)
    # 收到命令的时间、执行后的时间、pa pb pc
    # reply = {
    #     "robot_dof": robot.dof,
    #     "robot_deg": robot.degree
    # }
    # client_socket.send(json.dumps(reply).encode("utf-8"))

