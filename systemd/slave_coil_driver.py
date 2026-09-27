import sys

sys.path.append("/home/dengxutian/GitHub/2025_haptic_digital_twin-main")

try:
    import RPi.GPIO as GPIO
except:
    print("This system does not have RPi.GPIO package.")

import json
import socket
from utils.utils_hardware import Coil
LOCAL_IP = "192.168.137.2"
LOCAL_PORT = 12301

class SlaveRobot:

    def __init__(self):
        self.coil_a, self.loc_a = Coil(32, 33, 13), 45
        self.coil_b, self.loc_b = Coil(31, 29, 15), 135
        self.coil_c, self.loc_c = Coil(36, 38, 16), 225
        self.coil_d, self.loc_d = Coil(37, 40, 18), 315

        self.zero()

    def __del__(self):
        del self.coil_a
        del self.coil_b
        del self.coil_c
        del self.coil_d

        GPIO.cleanup()

    def zero(self):
        self.coil_a.set(0)
        self.coil_b.set(0)
        self.coil_c.set(0)
        self.coil_d.set(0)

    def to(self, percent_a, percent_b, percent_c, percent_d):

        print(f"Setting coils for percent: {percent_a}, {percent_b}, {percent_c}, {percent_d} ")
        # 正时电磁铁与永磁铁相吸
        # 负时电磁铁与永磁铁相斥
        self.coil_a.set(percent_a)
        self.coil_b.set(percent_b)
        self.coil_c.set(percent_c)
        self.coil_d.set(percent_d)

if __name__ == "__main__":

    server_socket = socket.socket()
    server_socket.bind((LOCAL_IP, LOCAL_PORT))  # 设置端口
    server_socket.listen(5)  # 设置排队数
    print(f"Driver is listening at {LOCAL_IP}:{LOCAL_PORT}")

    robot = SlaveRobot()
    client_socket, client_address = server_socket.accept()
    print(f"Driver is connected to {client_address[0]}:{client_address[1]}")

    while True:
        raw = client_socket.recv(1024)
        if not raw:
            break

        try:
            msg = json.loads(raw.decode("utf-8"))
            pa = float(msg["percent_a"])
            pb = float(msg["percent_b"])
            pc = float(msg["percent_c"])
            pd = float(msg["percent_d"])
        except (ValueError, KeyError) as e:
            print("接收到的消息格式错误：", e, raw)
            continue

        print(f"设置线圈 PWM 百分比：A={pa}, B={pb}, C={pc}, D={pd}")
        robot.to(pa, pb, pc, pd)


