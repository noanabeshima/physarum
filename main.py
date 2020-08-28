import numpy as np
from scipy import signal
import cv2
import time
from copy import deepcopy
from joblib import Parallel, delayed


class Angle:
    def __init__(self, angle = False):
        # If angle is False, generate random angle, else use angle from arg
        if angle is False:
            self.angle = 2*np.pi*np.random.random()
        else:
            angle = angle % (2*np.pi)
            self.angle = angle
    def unit(self):
        return np.array([-np.sin(self.angle), np.cos(self.angle)])
    def __add__(self, other):
        if isinstance(other, Angle):
            return Angle(self.angle + other.angle)
        else:
            return Angle(self.angle + other)
    def __sub__(self, other):
        if isinstance(other, Angle):
            return Angle(self.angle - other.angle)
        else:
            return Angle(self.angle - other)
    def __mul__(self, other):
        if isinstance(other, Angle):
            return Angle(self.angle*other.angle)
        else:
            return Angle(self.angle*other)
    def __radd__(self, other):
        return self.__add__(other)
    def __rsub__(self, other):
        return other.__sub__(self)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __str__(self):
        return str(self.angle)

class Agent:
    def __init__(self, world, sensor_angle=.4, sensor_offset=6, rotation_angle=.3, step_size=3, pos=False, orient=False):
        self.world = world
        if pos is False: # not initialized with a position
            self.position = (np.random.random(2)*np.array(self.world.shape))
            self.position = np.mod(self.position, np.array(self.world.shape)) # Make sure position fits in world
        else:
            self.position = pos
        if orient is False: # not initialized with an orientation/angle
            self.orientation = Angle() # Orientation in radians
        else:
            self.orientation = orient
        self.sensor_angle = Angle(sensor_angle)
        self.sensor_offset = sensor_offset
        self.rotation_angle = Angle(rotation_angle)
        self.step_size=step_size
    
    def step(self):        
        self.reorient()
        self.position = np.mod((self.position + self.step_size*self.orientation.unit()), self.world.shape)
        self.deposit()
        
    def reorient(self):
        orientations = [self.orientation + self.sensor_angle, self.orientation, self.orientation - self.sensor_angle]
        sample_coords = [self.position + self.sensor_offset*o.unit() for o in orientations]
        # Fix coords to the grid
        sample_coords = [np.mod(c.astype(int), self.world.shape) for c in sample_coords]
        
        # Get trail map values
        left, center, right = (self.world.grid[c[0],c[1]] for c in sample_coords)
        if center > left and center > right:
            # Stay facing the same direction
            pass
        elif center < left and center < right:
            # Rotate randomly left or right
            self.orientation = (self.orientation + .5*(np.random.random()+1)*np.random.choice([1,-1])*self.rotation_angle)
        elif left < right:
            # Rotate right
            self.orientation = self.orientation - .5*(np.random.random()+1)*self.rotation_angle
        elif right < left:
            # Rotate left
            self.orientation = self.orientation + .5*(np.random.random()+1)*self.rotation_angle
        else:
            pass

    def deposit(self):
        # Deposit 'trail chemical' onto the world grid
        pos = self.position.astype(int)
        self.world.grid[pos[0], pos[1]] += .005
        self.world.grid[pos[0], pos[1]].clip(0,1)


class World:
    def __init__(self, shape):
        self.shape = shape
        self.grid = np.zeros(shape)
        self.diffusion_kernel = np.array([[1, 1, 1],
                                          [1, 0, 1],
                                          [1, 1, 1]]).astype(np.float32)
        self.diffusion_kernel = .99*self.diffusion_kernel/self.diffusion_kernel.sum()
                                          
        self.agents = [Agent(self) for _ in range(1000)]

    def render(self):
        im = 255*np.array(self.grid)
        # im = np.kron(im, np.ones((2,2))) # Option to have 2x2 pixels per each grid cell
        cv2.imshow('Physarum', im)
        if cv2.waitKey(1) == 27: # Escape button will leave the video
            exit()

    def step(self):
        
        for agent in self.agents:
            agent.step()
        self.diffuse()

    def diffuse(self):
        self.grid = cv2.filter2D(self.grid, -1, self.diffusion_kernel)

def main():
    print("Press 'ESC' to end the program")
    world = World((500,500))

    while True:
        world.render()
        world.step()

if __name__=='__main__':
    main()
