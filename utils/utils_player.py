import numpy as np

def clip_deg(deg):
    
    while deg >= 360:
        deg -= 360
    while deg <= 0:
        deg += 360
    
    return deg


def is_number_line(line):
    
    line = line.strip()
    parts = line.split("，")
    for part in parts:
        try:
            float(part)
        except ValueError:
            return False
    return True


def load_txt(txt_path):
    
    good_lines = []
    with open(txt_path, "r") as file:
        lines = file.readlines()
    for line in lines:
        if is_number_line(line):
            good_lines.append(np.fromstring(line.replace("，", ","), sep=","))
    return np.array(good_lines)


# 这段代码可能需要修改
class Real2SimPlayer():
    
    def __init__(
        self,
        txt_path,
    ):
        
        self.data = load_txt(txt_path)
        self.txt_path = txt_path
        self.ind = 0
        self.done = False

    def step(self):
        
        if self.done:
            deg = None
            return deg, self.done
        
        vec = self.data[self.ind]
        x = vec[4]
        y = vec[5]
        
        rad = np.arctan2(y, x)
        deg = np.degrees(rad)
        deg = float(deg)
        # deg += 270
        # deg = clip_deg(deg)

        self.ind += 1
        self.done = (self.ind >= len(self.data))
        
        return deg, self.done


class SimpleReal2SimPlayer():
    
    def __init__(
        self,
        txt_path,
    ):
        
        self.data = np.loadtxt(txt_path)
        self.txt_path = txt_path
        self.ind = 0
        self.done = False

    def step(self):
        
        if self.done:
            deg = None
            return deg, self.done
        
        deg = self.data[self.ind]
        deg = float(deg)
        deg = deg

        self.ind += 1
        self.done = (self.ind >= len(self.data))
        
        return deg, self.done
    
    
class TestPlayer():
    
    def __init__(
        self,
    ):
        self.deg = 0

    def step(self):
        
        self.deg += 1
        
        return self.deg, False
