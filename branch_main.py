# 分叉位置调整电磁线圈减少碰撞
import threading
import time
import sys

from utils.utils_controller import EasyContoller, SlaveController

import keyboard
LOCAL_IP = "192.168.137.1"
LOCAL_PORT = 12202
ROBOT_IP = "192.168.137.3"
ROBOT_PORT = 12301

percent: list[float] = [0.0, 0.0, 0.0]
percent_d: float = 0.0
percent_lock = threading.Lock()
stop_event = threading.Event()

def easy_control():
    global percent, ROBOT_IP
    ROBOT_IP = "192.168.137.3"
    easy = EasyContoller(LOCAL_IP, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    while not stop_event.is_set():
        time.sleep(0.1)
        # read percent under lock and use it
        with percent_lock:
            p0, p1, p2 = percent[0], percent[1], percent[2]
        easy.step(p0, p1, p2)

def slave_control():
    global percent, percent_d
    slave = SlaveController(LOCAL_IP, LOCAL_PORT, ROBOT_IP, ROBOT_PORT)
    while not stop_event.is_set():
        time.sleep(0.1)
        # read percent under lock and use it
        with percent_lock:
            p0, p1, p2, p3 = percent[0], percent[1], percent[2], percent_d
        slave.step(p0, p1, p2, p3)


def _clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def _print_percent():
    with percent_lock:
        p0, p1, p2 = percent[0], percent[1], percent[2]
    print(f"\rpercent = [{p0:.1f}, {p1:.1f}, {p2:.1f}] ")


def _adjust(idx: int, delta: float):
    with percent_lock:
        percent[idx] = _clamp(percent[idx] + delta)
    _print_percent()


def _help_text():
    return ("Controls:\n"
            "  a/z -> +/ - percent[0]\n"
            "  s/x -> +/ - percent[1]\n"
            "  d/c -> +/ - percent[2]\n"
            "  h   -> help\n"
            "  q   -> quit\n")


def keyboard_listener_fallback(event):
    global percent, percent_d
    if event.name == 'a':
        _adjust(0, +0.1)
    elif event.name == 'z':
        _adjust(0, -0.1)
    elif event.name == 's':
        _adjust(1, +0.1)
    elif event.name == 'x':
        _adjust(1, -0.1)
    elif event.name == 'd':
        _adjust(2, +0.1)
    elif event.name == 'c':
        _adjust(2, -0.1)
    elif event.name == 'h':
        print('\n' + _help_text(), end='')
    # elif event.name == 'e':
    #     percent = [0.0, 0.0, 0.0]
    # elif event.name == 'q':
    #     percent = [1.0, 0.0, 0.0]
    # elif event.name == 'w':
    #     percent = [0.0, 1.0, 0.0]
    elif event.name == 'f':
        if percent_d + 0.1 <=1:
            percent_d = percent_d + 0.1
    elif event.name == 'v':
        if percent_d - 0.1 >=-1:
            percent_d = percent_d - 0.1
    elif event.name == 'l':
        percent = [0.0, 0.0, -0.8]
        print('set to left')
    elif event.name == 'r':
        percent = [-1.0, 0.2, -0.8]
        print('set to right')
    elif event.name == 'q':
        print('\nQuitting...')
        stop_event.set()


if __name__ == '__main__':
    # start easy control in background
    keyboard.on_press(keyboard_listener_fallback)

    t_easy = threading.Thread(target=easy_control)
    t_easy.start()

    t_easy.join()
    # t_slave = threading.Thread(target=slave_control)
    # t_slave.start()
    # t_slave.join()
    keyboard.wait('esc')


#[0,0,-0.8] left
#[-1,0.2,-0.8] right
"""
sudo /home/xwj/anaconda3/envs/mygenesis/bin/python3.11 /home/xwj/桌面/project_2026/branch_main.py 
"""