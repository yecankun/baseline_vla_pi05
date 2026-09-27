# import math
# import time
#
# try:
#     import RPi.GPIO as GPIO
# except:
#     print("This system does not have RPi.GPIO package.")
#
# class Coil:
#
#     def __init__(self, IN1, IN2, EN, Freq=5000):
#
#         GPIO.setmode(GPIO.BOARD)
#         GPIO.setwarnings(False)
#
#         GPIO.setup([IN1, IN2, EN], GPIO.OUT)
#         self.PWM = GPIO.PWM(EN, Freq)
#         self.IN1, self.IN2, self.EN = IN1, IN2, EN
#
#         self.PWM.start(0)
#
#     def __del__(self):
#
#         GPIO.cleanup()
#
#     def set(self, current):
#
#         assert current >= -1 and current <= 1
#
#         if current > 0:
#             GPIO.output(self.IN1, GPIO.HIGH)
#             GPIO.output(self.IN2, GPIO.LOW)
#             self.PWM.ChangeDutyCycle(current*100)
#
#         elif current < 0:
#             GPIO.output(self.IN1, GPIO.LOW)
#             GPIO.output(self.IN2, GPIO.HIGH)
#             self.PWM.ChangeDutyCycle(-current*100)
#
#         else:
#             GPIO.output(self.IN1, GPIO.LOW)
#             GPIO.output(self.IN2, GPIO.LOW)
#             self.PWM.ChangeDutyCycle(0)
#
#
# class Robot():
#
#     def __init__(self):
#
#         self.coil_a, self.loc_a = Coil(31, 32, 29), 30
#         self.coil_b, self.loc_b = Coil(35, 36, 33), 150
#         self.coil_c, self.loc_c = Coil(38, 40, 37), 270
#
#         self.zero()
#
#     def __del__(self):
#
#         del self.coil_a
#         del self.coil_b
#         del self.coil_c
#
#         GPIO.cleanup()
#
#     def zero(self):
#
#         self.degree = None
#         self.dof = [0, 0, 0]
#         self.coil_a.set(0)
#         self.coil_b.set(0)
#         self.coil_c.set(0)
#
#     def to(self, force):
#         #
#         # # self.degree = degree
#         # self.dof = [
#         #     math.cos(math.radians(self.degree - self.loc_a)),  # 余弦值
#         #     math.cos(math.radians(self.degree - self.loc_b)),
#         #     math.cos(math.radians(self.degree - self.loc_c)),
#         # ]
#
#         # self.coil_a.set(self.dof[0])
#         # self.coil_b.set(self.dof[1])
#         # self.coil_c.set(self.dof[2])
#         print(f"Setting coils for force: {force}")
#         # # 正时电磁铁与永磁铁相吸
#         # self.coil_a.set(1)
#         # self.coil_b.set(1)
#         # self.coil_c.set(1)
#
#         # 负时电磁铁与永磁铁相斥
#         self.coil_a.set(1)
#         self.coil_b.set(1)
#         self.coil_c.set(1)




import math
import time

try:
    import RPi.GPIO as GPIO
except:
    print("This system does not have RPi.GPIO package.")


class Coil:

    def __init__(self, IN1, IN2, EN, Freq=5000):

        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        GPIO.setup([IN1, IN2, EN], GPIO.OUT)
        self.PWM = GPIO.PWM(EN, Freq)
        self.IN1, self.IN2, self.EN = IN1, IN2, EN

        self.PWM.start(0)

    def __del__(self):

        GPIO.cleanup()

    def set(self, current):

        assert current >= -1 and current <= 1

        if current > 0:
            GPIO.output(self.IN1, GPIO.HIGH)
            GPIO.output(self.IN2, GPIO.LOW)
            self.PWM.ChangeDutyCycle(current * 100)

        elif current < 0:
            GPIO.output(self.IN1, GPIO.LOW)
            GPIO.output(self.IN2, GPIO.HIGH)
            self.PWM.ChangeDutyCycle(-current * 100)

        else:
            GPIO.output(self.IN1, GPIO.LOW)
            GPIO.output(self.IN2, GPIO.LOW)
            self.PWM.ChangeDutyCycle(0)


class Robot():

    def __init__(self):
        self.coil_a, self.loc_a = Coil(31, 32, 29), 30
        self.coil_b, self.loc_b = Coil(35, 36, 33), 150
        self.coil_c, self.loc_c = Coil(38, 40, 37), 270

        self.zero()

    def __del__(self):
        del self.coil_a
        del self.coil_b
        del self.coil_c

        GPIO.cleanup()

    def zero(self):
        self.degree = None
        self.dof = [0, 0, 0]
        self.coil_a.set(0)
        self.coil_b.set(0)
        self.coil_c.set(0)

    # def to(self, percent_a, percent_b, percent_c):
    #     #
    #     # # self.degree = degree
    #     # self.dof = [
    #     #     math.cos(math.radians(self.degree - self.loc_a)),  # 余弦值
    #     #     math.cos(math.radians(self.degree - self.loc_b)),
    #     #     math.cos(math.radians(self.degree - self.loc_c)),
    #     # ]
    #
    #     # self.coil_a.set(self.dof[0])
    #     # self.coil_b.set(self.dof[1])
    #     # self.coil_c.set(self.dof[2])
    #     print(f"Setting coils for percent: {percent_a}, {percent_b}, {percent_c}")
    #     # # 正时电磁铁与永磁铁相吸
    #     # self.coil_a.set(1)
    #     # self.coil_b.set(1)
    #     # self.coil_c.set(1)
    #
    #     # 负时电磁铁与永磁铁相斥
    #     self.coil_a.set(percent_a)
    #     self.coil_b.set(percent_b)
    #     self.coil_c.set(percent_c)

    def to(self, percent_a, percent_b, percent_c):
        #
        # # self.degree = degree
        # self.dof = [
        #     math.cos(math.radians(self.degree - self.loc_a)),  # 余弦值
        #     math.cos(math.radians(self.degree - self.loc_b)),
        #     math.cos(math.radians(self.degree - self.loc_c)),
        # ]

        # self.coil_a.set(self.dof[0])
        # self.coil_b.set(self.dof[1])
        # self.coil_c.set(self.dof[2])
        print(f"Setting coils for percent: {percent_a}, {percent_b}, {percent_c}")
        # # 正时电磁铁与永磁铁相吸
        # self.coil_a.set(1)
        # self.coil_b.set(1)
        # self.coil_c.set(1)

        # 负时电磁铁与永磁铁相斥
        self.coil_a.set(percent_a)
        self.coil_b.set(percent_b)
        self.coil_c.set(percent_c)