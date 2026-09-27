# import json
# import socket
#
# # def clip_deg(deg):
# #
# #     assert type(deg) in [int, float]
# #     while deg > 360:
# #         deg = deg - 360
# #     while deg < 0:
# #         deg = deg + 360
# #     return deg
#
# class EasyContoller:
#
#     def __init__(
#         self,
#         LOCAL_IP,
#         LOCAL_PORT,
#         ROBOT_IP,
#         ROBOT_PORT,
#         ):
#
#         self.client_socket = socket.socket()
#         print(f"Local Client is bonding to {LOCAL_IP}:{LOCAL_PORT}")
#         self.client_socket.bind((LOCAL_IP, LOCAL_PORT))
#         print(f"Local Client is connecting to {ROBOT_IP}:{ROBOT_PORT}")
#         self.client_socket.connect((ROBOT_IP, ROBOT_PORT))
#
#     def step(self, force: float):
#
#         # deg = clip_deg(deg)
#         send_msg = str(force).encode("utf-8")
#         self.client_socket.send(send_msg)
#
#         recv_msg = self.client_socket.recv(1024)
#         recv_msg = json.loads(recv_msg.decode("utf-8"))
#         return recv_msg


import json
import socket
import time


# def clip_deg(deg):
#
#     assert type(deg) in [int, float]
#     while deg > 360:
#         deg = deg - 360
#     while deg < 0:
#         deg = deg + 360
#     return deg

class EasyContoller:

    def __init__(
            self,
            LOCAL_IP,
            LOCAL_PORT,
            ROBOT_IP,
            ROBOT_PORT,
    ):
        self.client_socket = socket.socket()
        print(f"Local Client is bonding to {LOCAL_IP}:{LOCAL_PORT}")
        self.client_socket.bind((LOCAL_IP, LOCAL_PORT))
        print(f"Local Client is connecting to {ROBOT_IP}:{ROBOT_PORT}")
        self.client_socket.connect((ROBOT_IP, ROBOT_PORT))

    def step(self, percent_a: float, percent_b: float, percent_c: float):
        # 1) 组装 JSON 并发送
        payload = {
            "percent_a": percent_a,
            "percent_b": percent_b,
            "percent_c": percent_c
        }
        self.client_socket.send(json.dumps(payload).encode("utf-8"))
        # 无需接受服务器端回复

class SlaveController:
    def __init__(
            self,
            LOCAL_IP,
            LOCAL_PORT,
            ROBOT_IP,
            ROBOT_PORT,
    ):
        self.client_socket = socket.socket()
        print(f"Local Client is binding to {LOCAL_IP}:{LOCAL_PORT}")
        self.client_socket.bind((LOCAL_IP, LOCAL_PORT))
        print(f"Local Client is connecting to {ROBOT_IP}:{ROBOT_PORT}")
        self.client_socket.connect((ROBOT_IP, ROBOT_PORT))

    def step(self, percent_a: float, percent_b: float, percent_c: float, percent_d: float):
        # 1) 组装 JSON 并发送
        payload = {
            "percent_a": percent_a,
            "percent_b": percent_b,
            "percent_c": percent_c,
            "percent_d": percent_d
        }
        self.client_socket.send(json.dumps(payload).encode("utf-8"))

#主函数
if __name__ == "__main__":
    LOCAL_IP = "192.168.137.201"
    LOCAL_PORT = 12201
    ROBOT_IP = "192.168.137.3"
    ROBOT_PORT = 12301
    easy = EasyContoller(LOCAL_IP,LOCAL_PORT,ROBOT_IP,ROBOT_PORT)
    while True:
        time.sleep(0.1)
        easy.step(1, 0, 0)
        print('---')
