import os

os.environ['PYOPENGL_PLATFORM'] = 'glx'
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from stl import mesh
import trimesh
model_path= './model/0422.stl'
#trimesh_model = trimesh.load(model_path)

class Interface:
    def __init__(self, model_path='utils/interface/model/0422.stl'):
        self.IS_PERSPECTIVE = False
        self.VIEW = np.array([-1.5, 1.5, -1.5, 1.5, 0.5, 50.0])
        self.SCALE_K = np.array([1.0, 1.0, 1.0])
        self.EYE = np.array([0.0, 0.0, 10.0])
        self.LOOK_AT = np.array([0.0, 0.0, 0.0])
        self.EYE_UP = np.array([0.0, 1.0, 0.0])
        self.WIN_W, self.WIN_H = 1080, 960
        self.LEFT_IS_DOWNED = False
        self.MOUSE_X, self.MOUSE_Y = 0, 0
        self.sphere_position = np.array([0.1, 0.1, -0.4])
        self.target_position = np.array([-0.7, 3.7, -0.1])
        self.move_step = 0.1
        self.model_path = model_path
        self.piper_order = 0
        self.camera_pos = []
        self.DIST, self.PHI, self.THETA = self.getposture()
        self.VIEW_MODE = 0  # 0-自由视角, 1-XY平面, 2-YZ平面, 3-XZ平面

    def getposture(self):
        dist = np.linalg.norm(self.EYE - self.LOOK_AT)
        if dist > 0:
            phi = np.arcsin((self.EYE[1] - self.LOOK_AT[1]) / dist)
            theta = np.arcsin((self.EYE[0] - self.LOOK_AT[0]) / (dist * np.cos(phi)))
        else:
            phi, theta = 0.0, 0.0
        return dist, phi, theta

    def change_pos(self, new_pos):
        self.sphere_position[0] = new_pos[0]
        self.sphere_position[1] = new_pos[1]
        self.sphere_position[2] = new_pos[2]

    def update_with_transform(self, real_coord):
        self.sphere_position[0] = real_coord[2] * 10 - 4
        self.sphere_position[1] = -real_coord[0] * 10 + 0.73 + 0.9
        self.sphere_position[2] = -real_coord[1] * 10 + 0.03


    def init(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)  # 启用光源 0
        glShadeModel(GL_SMOOTH)

        # 设置光源属性（光源 0）
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 10, 1])  # 正前方光源
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])  # 全强度白光
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])  # 微弱环境光

        # 设置材质属性
        glMaterialfv(GL_FRONT, GL_DIFFUSE, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))  # 设置物体的漫反射颜色
        glMaterialfv(GL_FRONT, GL_SPECULAR, np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))  # 设置物体的高光颜色
        glMaterialfv(GL_FRONT, GL_SHININESS, np.array([50.0], dtype=np.float32))  # 设置物体的高光强度

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)  # 更精确的深度比较

    def load_stl(self, file_path):
        return mesh.Mesh.from_file(file_path)

    def draw_stl(self, model):
        glEnable(GL_NORMALIZE)  # 处理法线长度

        if model is None:
            return

        # 启用颜色材质
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # 启用混合（实现透明）
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)  # 仍然需要深度测试
        glDepthMask(GL_FALSE)  # 关闭深度写入，避免影响后面的物体

        glColor4f(0.8, 0.2, 0.2, 0.1)  # 半透明暗红色

        # 使用顶点数组
        vertices = model.vectors.reshape(-1, 3)
        normals = np.repeat(model.normals, 3, axis=0)

        # 批量绘制
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glNormalPointer(GL_FLOAT, 0, normals)
        glDrawArrays(GL_TRIANGLES, 0, len(vertices))

        # 清理状态
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisable(GL_NORMALIZE)

        # 关闭混合
        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)  # 重新开启深度写入

        glDisable(GL_COLOR_MATERIAL)

    def draw_sphere(self):
        glPushMatrix()
        glTranslatef(self.sphere_position[0], self.sphere_position[1], self.sphere_position[2])
        glEnable(GL_COLOR_MATERIAL)
        glColor3f(0.0, 1.0, 0.0)
        glutSolidSphere(0.1, 20, 20)
        glPopMatrix()
        glDisable(GL_COLOR_MATERIAL)

    def draw_target(self):
        glPushMatrix()
        glTranslatef(self.target_position[0], self.target_position[1], self.target_position[2])
        glEnable(GL_COLOR_MATERIAL)
        glColor3f(0.0, 0.0, 1.0)
        glutSolidSphere(0.1, 20, 20)
        glPopMatrix()
        glDisable(GL_COLOR_MATERIAL)

    def draw(self):
        global EYE, LOOK_AT, EYE_UP
        global IS_PERSPECTIVE, VIEW
        global SCALE_K
        global WIN_W, WIN_H

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 主视角占据左半屏
        glViewport(0, 0, self.WIN_W * 3 // 4, self.WIN_H)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.WIN_W / self.WIN_H, 0.1, 500)
        gluLookAt(self.EYE[0], self.EYE[1], self.EYE[2], self.LOOK_AT[0], self.LOOK_AT[1], self.LOOK_AT[2], self.EYE_UP[0], self.EYE_UP[1], self.EYE_UP[2])
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.draw_sphere()
        self.draw_target()
        stl_mesh = self.load_stl(self.model_path)
        self.draw_stl(stl_mesh)

        # 右侧三视图
        view_width = self.WIN_W // 4  # 缩小宽度
        view_height = self.WIN_H // 3  # 三分之一高度

        # XY平面
        glViewport(self.WIN_W * 3 // 4, self.WIN_H * 2 // 3, view_width, view_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-4, 4, -4, 4, -500, 500)
        gluLookAt(0, 0, 200, 0, 0, 0, 0, 1, 0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.draw_sphere()
        self.draw_target()
        self.draw_stl(stl_mesh)

        # YZ平面
        glViewport(self.WIN_W * 3 // 4, self.WIN_H // 3, view_width, view_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-4, 4, -4, 4, -500, 500)
        gluLookAt(200, 0, 0, 0, 0, 0, 0, 0, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.draw_sphere()
        self.draw_target()
        self.draw_stl(stl_mesh)

        # XZ平面
        glViewport(self.WIN_W * 3 // 4, 0, view_width, view_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-4, 4, -4, 4, -500, 500)
        gluLookAt(0, 200, 0, 0, 0, 0, 1, 0, 0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.draw_sphere()
        self.draw_target()
        self.draw_stl(stl_mesh)

        glutSwapBuffers()

    def keydown(self, key, x, y):
        global target_position
        global DIST, PHI, THETA
        global EYE, LOOK_AT, EYE_UP
        global IS_PERSPECTIVE, VIEW

        global piper_order
        key = key.decode('utf-8').lower()
        # 添加视图切换功能
        if key == '1':
            self.set_view(1)  # XY平面
        elif key == '2':
            self.set_view(2)  # YZ平面
        elif key == '3':
            self.set_view(3)  # XZ平面
        elif key == '0':
            self.set_view(0)  # 自由视角

        if key == 'w':
            self.target_position[1] += self.move_step
            print(f'target: {self.target_position}')
        elif key == 's':
            self.target_position[1] -= self.move_step
            print(f'target: {self.target_position}')
        elif key == 'a':
            self.target_position[0] -= self.move_step
            print(f'target: {self.target_position}')
        elif key == 'd':
            self.target_position[0] += self.move_step
            print(f'target: {self.target_position}')
            #print('inside: ', trimesh_model.contains([self.target_position]))
        elif key == 'q':
            self.target_position[2] += self.move_step
            print(f'target: {self.target_position}')
        elif key == 'e':
            self.target_position[2] -= self.move_step
            print(f'target: {self.target_position}')
        elif key == 'l':
            piper_order = 1
            print('piper_order')
        glutPostRedisplay()

    def mouseclick(self, button, state, x, y):

        self.MOUSE_X, self.MOUSE_Y = x, y
        if button == GLUT_LEFT_BUTTON:
            self.LEFT_IS_DOWNED = state == GLUT_DOWN
        elif button == 3:
            self.SCALE_K *= 1.05
            glutPostRedisplay()
        elif button == 4:
            self.SCALE_K *= 0.95
            glutPostRedisplay()

    def mousemotion(self, x, y):
        if self.LEFT_IS_DOWNED:
            dx = self.MOUSE_X - x
            dy = y - self.MOUSE_Y
            self.MOUSE_X, self.MOUSE_Y = x, y

            self.PHI += 2 * np.pi * dy / WIN_H
            self.PHI %= 2 * np.pi
            self.THETA += 2 * np.pi * dx / WIN_W
            self.THETA %= 2 * np.pi
            r = DIST * np.cos(self.PHI)

            self.EYE[1] = DIST * np.sin(self.PHI)
            self.EYE[0] = r * np.sin(self.THETA)
            self.EYE[2] = r * np.cos(self.THETA)

            if 0.5 * np.pi < self.PHI < 1.5 * np.pi:
                self.EYE_UP[1] = -1.0
            else:
                self.EYE_UP[1] = 1.0

            glutPostRedisplay()

    def set_view(self, mode):
        self.VIEW_MODE = mode

        # 保持当前观察距离
        dist = np.linalg.norm(self.EYE - self.LOOK_AT)

        if mode == 1:  # XY平面
            self.EYE = np.array([0, 0, dist])
            self.LOOK_AT = np.array([0, 0, 0])
            self.EYE_UP = np.array([0, 1, 0])
        elif mode == 2:  # YZ平面
            self.EYE = np.array([dist, 0, 0])
            self.LOOK_AT = np.array([0, 0, 0])
            self.EYE_UP = np.array([0, 0, 1])
        elif mode == 3:  # XZ平面
            self.EYE = np.array([0, dist, 0])
            self.LOOK_AT = np.array([0, 0, 0])
            self.EYE_UP = np.array([1, 0, 0])
        else:  # 自由视角
            self.EYE = np.array([0, 0, dist])
            self.LOOK_AT = np.array([0, 0, 0])
            self.EYE_UP = np.array([0, 1, 0])

        # 重置视角参数
        self.DIST, self.PHI, self.THETA = self.getposture()
        glutPostRedisplay()

    def reshape(self, width, height):
        self.WIN_W, self.WIN_H = width, height

    def idle(self):
        glutPostRedisplay()

    def opengl_main(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
        glutInitWindowSize(self.WIN_W, self.WIN_H)
        glutInitWindowPosition(300, 200)
        glutCreateWindow(b'OpenGL Sphere & STL')
        self.init()
        glutDisplayFunc(self.draw)
        glutReshapeFunc(self.reshape)  # 注册响应窗口改变的函数reshape()
        glutMouseFunc(self.mouseclick)  # 注册响应鼠标点击的函数mouseclick()
        glutMotionFunc(self.mousemotion)
        glutKeyboardFunc(self.keydown)

        glutIdleFunc(self.idle)
        glutMainLoop()

if __name__ == "__main__":

    interface = Interface(model_path='./model/0422.stl')
    interface.opengl_main()
