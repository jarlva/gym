# import math
# https://github.com/jinfagang/rl-solution/blob/master/solve_cart_pole.py
import gym
from gym import spaces #, logger
from gym.utils import seeding
import numpy as np
import random
from sklearn.preprocessing import minmax_scale

class MyCartPoleEnv(gym.Env):
    metadata = { 'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50 }

    def __init__(self):
        file = r'C:\Users\Jake\py\Files\Keras-RL\sine.csv'
        self.data = np.loadtxt(file, skiprows=1 , delimiter='\t', dtype=np.float64, usecols=7, )
        self.data = minmax_scale(self.data, feature_range=(-1.0, 1.0), copy= False) # Normalize data -1,1. https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
        self.cycle = 255 # how many bars per episode, increasing from 170 to 255 rose score faster.
        self.prevbars = 250 # how many previous bars to include. increasing number increased rose score faster.
        low = np.concatenate(
            (np.full((self.prevbars), -1.0),    # reward
            [0,                                 # bar
            0]))                                # alreadybought
        high = np.concatenate(                  # close
            (np.full((self.prevbars), 1.0),     # reward 
            [self.cycle+1,                      # bar
            1]))                                # alreadybought

        # low = low.astype(np.float64, copy=True)
        # high = high.astype(np.float64, copy=True)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.action_space = spaces.Discrete(3)  # noop/buy/sell
        self.reward_range = (-1.0, 1.0)
        self.seed()
        self.viewer = None
        self.state = None
        self.okrender = False
        self.ok_to_render = False # if okrender true will be activated after X steps
        self.stepcount = 0
        self.bar =      0
        # setup plot      
        self.screen_height =200 #200 #348 # 100 dots per inch
        self.screen_width = self.cycle-1 #400 #64 #400 #678
        self.seed()
        self.reset()

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
            reward = 0.0001
            # if self.ok_to_render: self.ax.plot(self.bar, self.Y,'r,') 

        # Valid sell
        elif action == 2 and self.already_bought: 
            self.done = True
            reward = ( self.Y - self.price ) / 2    # sine -1,1
            # reward = np.clip( ( self.Y - self.price ) / 2, -0.997, 0.997)    # sine -1,1
            # tmp =  ( ( self.Y - self.price ) / self.price ) /100  # sigmoid from -5 to 5
            # reward = np.clip(tmp, -1.0, 1.0 )
            # reward = ( self.Y - self.price ) / 10  # sine 0,10
            # if self.Y - self.price > 0.5:
            # if reward > 0.2:
            #     reward = 0.99
            # else:
            #      reward = -1
            if self.ok_to_render: 
                self.ax.plot(self.bar, self.Y, 'b.') 
                print(' rw:', reward, self.Y, self.price)

        # Price change after buy 
        # For pure sine: initialy adversly impact score. but after 2k it peaks higher  (0.96) than without it
        elif self.already_bought: 
            if self.Y > self.prev1: # punish/reward based on price change
                reward = 0.0001
            else:
                reward = -0.0001
            if self.ok_to_render: self.ax.plot(self.bar, self.Y,'y,')

        # incentive to bottom
        elif not self.already_bought and self.Y < self.prev1:
            reward = 0.0001
            
        # else: 
            if self.ok_to_render: self.ax.plot(self.bar, self.Y,'w,') 

        if self.bar > self.cycle:
            self.done = True # for simulating end of episode
            # if reward == 0.0: # if self.already_bought : # end of chart, bought but did not sell
            reward = -0.95 # punish 

        # if reward > 1: print(' rw:', reward, self.Y, self.price)
        return self._get_ob(), reward, self.done, {"bar": self.bar} # , dtype=np.float32

    def _get_ob(self):
        state = np.concatenate(( self.data[self.X - self.prevbars+1:self.X+1] , (self.bar, self.already_bought)))
        return np.array(state) #, dtype=np.float32)

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
        # if self.okrender and self.stepcount < 200000:  #  1000 ~= 4,500 steps, every 2.1  steps
            self.ax.clear()
            self.ax.set_xlim(0, self.screen_width )
            self.ax.set_ylim(-1.01, 1.01)
            self.ax.axis('off')
        return self._get_ob()
        
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
        img = buf.reshape((self.screen_height, self.screen_width, 3)) 
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