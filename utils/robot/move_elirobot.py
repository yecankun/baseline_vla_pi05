from elite import EC
ec = EC(ip="192.168.137.200", auto_connect=True)

import linecache
def write_in(file_name: str, data):
    with open(file_name, encoding="utf-8", mode="a") as file:
        file.write(str(data) + '\n')
POSE_PATH = '/home/xwj/桌面/project_2026/path/path2_pose1.txt'
while True:
    op=input('input:')
    pos=ec.current_pose
#相机侧视视角
#前后
    if op=='q':
        pos[0]+=5
    elif op=='a':
        pos[0]-=5
#左右
    elif op=='w':
        pos[1]+=15
    elif op=='s':
        pos[1]-=15
#上下
    elif op=='e':
        pos[2]+=5
    elif op=='d':
        pos[2]-=5
    #方位朝向
    #上
    elif op=='r':
        pos[3]+=0.1
    #下
    elif op=='f':
        pos[3]-=0.1
    elif op=='t':
        pos[4]+=0.1
    elif op=='g':
        pos[4]-=0.1
    elif op=='y':
        pos[5]+=0.1
    elif op=='h':
        pos[5]-=0.1
    elif op=='u':
        print('joint:')
        print(ec.current_joint)
        print('pose:')
        print(ec.current_pose)
    elif op=='j':
        cur_pose = ec.current_pose
        write_in(POSE_PATH,f"{cur_pose[0]} {cur_pose[1]} {cur_pose[2]} {cur_pose[3]} {cur_pose[4]} {cur_pose[5]}")
        print(f'pose saved at {cur_pose}')
        #write_in('/home/xwj/桌面/genesis/path/path1_joint.txt',ec.current_joint)
    elif op == 'm':
        target_pos=[-518.0865471778408, 241.89690160238519, 460.52756627772334, -3.092430608400453, 0.017300840263059253,
     0.16063413523099443]
        target=ec.get_inverse_kinematic(pose=target_pos)
        ec.move_joint(target_joint=target,speed=20)
    else:
        break
        #print('error')
    if ec.current_pose!=pos:
        target=ec.get_inverse_kinematic(pose=pos)
        ec.move_joint(target_joint=target,speed=20)