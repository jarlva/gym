import math
import gym
from gym import spaces #, logger
from gym.utils import seeding
import numpy as np
import random

class MyCartPoleEnv(gym.Env):
    
    metadata = { 'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50 }

    def __init__(self):
        file = r'C:\Users\Jake\py\Files\Keras-RL\sine.csv'
        self.data = np.loadtxt(file, skiprows=1 , delimiter='\t', dtype=np.float64, usecols=7, )
        self.cycle = 170 # how many bars per episode
        self.prevbars = 30 #62 # how many previous bars to include
        low = np.concatenate(
            (np.full((self.prevbars), -1.0),    # close
            [0,                                 # bar
            0]))                                # alreadybought
        high = np.concatenate(                  # close
            (np.full((self.prevbars), 1.0), 
            [self.cycle+2,                      # bar
            1]))                                # alreadybought

        low = low.astype(np.float32, copy=True)
        high = high.astype(np.float32, copy=True)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # noop/buy/sell
        self.reward_range = (-1., 1.)
        self.seed()
        self.viewer = None
        self.state = None
        self.okrender =  True
        self.ok_to_render = False # if okrender true will be activated after X steps
        self.stepcount = 0
        self.bar = 0
        # setup plot      
        self.screen_height =200 #200 #348 # 100 dots per inch
        self.screen_width = 400 #64 #400 #678
        self.seed()
        self.reset()

    def reset(self):
        self.stepcount += self.bar
        self.bar = 0
        self.done = False
        self.already_bought = False
        # self.amp = random.uniform(1.0,3.0)
        # self.freq = random.randint(3,9)
        # self.X =  random.randint(1, 360)
        # self.Y = np.sin(self.X/ self.factor) / self.amp
        self.X = random.randint(self.prevbars, int(self.data.size-self.cycle-2))
        self.Y = self.data[self.X]
        # plot reset
        if self.ok_to_render:
        # if self.okrender and self.stepcount < 200000 :  #  1000 ~= 4,500 steps, every 2.1  steps
            self.ok_to_render = True
            self.ax.clear()
            self.ax.set_xlim(1, self.screen_width)
            self.ax.set_ylim(-1.01, 1.01)
            self.ax.axis('off')
        return self._get_ob()

    def step(self, action):
        reward = 0.0
        self.prev1 = self.Y
        self.X += 1
        self.bar += 1
        
        # self.Y = np.sin(self.X/ self.factor) / self.amp
        self.Y = self.data[self.X]

        # Valid buy
        if action == 1 and not self.already_bought: 
            self.already_bought = True
            self.price = self.Y     # purchase price
            self.reward = 0.001
            # reward = (0 - self.Y)  
            # if self.ok_to_render: self.ax.plot(self.bar, self.Y,'r,') 

        # Valid sell
        elif action == 2 and self.already_bought: 
            self.done = True
            reward = ( self.Y - self.price ) / 2    # sigmoid from -5 to 5
            if self.Y - self.price > 0.5:
                reward = 1.0
            # else:
            #      reward = -1
            # if self.ok_to_render: self.ax.plot(self.bar, self.Y, 'b,') 

        # Price rise after buy
        # elif self.already_bought: 
        #     if self.Y > self.prev1: # punish/reward based on price change
        #         reward = 0.01
        #     else:
        #         reward = -0.01
        #     if self.ok_to_render: self.ax.plot(self.bar, self.Y,'w,')

        # else: 
        #     if self.ok_to_render: self.ax.plot(self.bar, self.Y,'w,') 

        if self.bar > self.cycle:
            self.done = True # for simulating end of episode
            # if reward == 0.0: # if self.already_bought : # end of chart, bought but did not sell
            reward = -0.9 # punish 

        return self._get_ob(), reward, self.done, {"bar": self.bar} # , dtype=np.float32

    def _get_ob(self):
        state = np.concatenate(( self.data[self.X - self.prevbars+1:self.X+1] , (self.bar, self.already_bought)))
        return np.array(state, dtype=np.float32)

    def render(self, mode='human'):
        # if self.ok_to_render: # render only after x steps
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            import matplotlib.pyplot as plt
            self.fig, self.ax = plt.subplots(squeeze=True)#, squeeze=True
            self.fig.set_size_inches( self.screen_width/100, self.screen_height/100)  # (w, h)
            self.fig.patch.set_facecolor('black')
            self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0) # https://code.i-harness.com/en/q/8dd4b2
            self.ok_to_render = True
            self.viewer = rendering.SimpleImageViewer()
        img = self._get_image() # img is a numpy x,y,3 array, 210,160,3
        self.viewer.imshow(img) # display the breakout inital screen
        return self.viewer.isopen

    def _get_image(self):
        # convert matplot to numpy array rgb and crop tight
        self.fig.canvas.draw()
        # print(self.fig.canvas.get_width_height())
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        # # print(buf.shape)
        # # ncols, nrows = self.fig.canvas.get_width_height()
        img = buf.reshape((self.screen_height, self.screen_width, 3)) # numpy, todo: need to crop it to reduce cpu overhead
        # # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        return img

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]