# winsound.Beep(900, 100)
'''
Self explanation: https://medium.com/quick-code/understanding-self-in-python-a3704319e5f0
      https://matplotlib.org/api/axes_api.html#axis-limits-and-direction
    https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure
    https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
'''
import numpy as np
import os
import gym
from gym import utils, error, spaces, logger
from gym.utils import seeding
import matplotlib.pyplot as plt
import random
from scipy.special import expit, logit # for zigmoid reward
# import winsound
# import io
# from PIL import Image
from gym.envs.classic_control import rendering

class MyEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, obs_type = 'image', frameskip = 4, repeat_action_probability=0.0):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""
        utils.EzPickle.__init__(self, frameskip, repeat_action_probability)
        self._observation = []
        # https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
        self.frameskip = 4 #frameskip force 4 frame skips
        self.done = False
        self._action_set = np.array( [0, 1, 2], dtype=int )  # noop/buy/sell
        self.action_space = spaces.Discrete(len(self._action_set))
        self.render_counter = 0
        self.ok_to_render = False
        self.false_buy = 0
        self.prevX = 0
        self.prevY = 0
        # setup plot      
        screen_height =64 #200 #348 # 100 dots per inch
        screen_width = 64 #400 #678
        self.fig, self.ax = plt.subplots(squeeze=True)#, squeeze=True
        self.fig.set_size_inches( screen_width/100, screen_height/100)  # (w, h)
        self.fig.patch.set_facecolor('black')
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0) # https://code.i-harness.com/en/q/8dd4b2
        self.ncols, self.nrows = self.fig.canvas.get_width_height()
        print('plot area:',self.ncols, self.nrows)
        # self.ax.set_xlim(1, 64)
        # self.ax.set_ylim(-1.05, 1.05)
        # self.ax.set_xbound(1, 64)
        # self.ax.set_ybound(-1.05, 1.05)
        # self.ax.autoscale(False)
        # self.fig.autoscale(False)
        # self.ax._autoscaleXon = False
        # self.ax._autoscaleYon = False
        # self.ax.set_aspect(1)
        # self.fig.set_size_inches( 0.84, 0.84)  # (w, h)

        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.seed()
        self.viewer = None
        self.state = None

    def reset(self):
        # action logic
        self.bar = 2
        self.done = False
        self.freq = random.randint(3,9)
        self.X = random.randint(1, 360)
        # self.amp = random.uniform(1.0,3.0)
        self.already_bought = False
        self.price = 0.0
        if self.render_counter > int( 100000 / 6):  # XXXXXXXX control when to start rendering 6000 ~= 40,000 steps, every 6.7 steps
            self.ok_to_render = True
        self.render_counter += 1

        # plot reset
        self.ax.clear()
        self.ax.set_xlim(1, 64)
        self.ax.set_ylim(-1.05, 1.05)
        self.ax.axis('off')
        # self.ax.autoscale(False)
        # self.fig.patch.set_facecolor('black')
        # print(' reset')
        # logger.info("reseting")
        return self._get_image()

    def render(self, mode='human'):
        # print ('', self.render_counter  )
        img = self._get_image() # img is a numpy x,y,3 array, 210,160,3
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
        # elif self.ok_to_render: # render only after x steps
            # from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img) # display the breakout inital screen
            return self.viewer.isopen

    def _get_image(self):
        # convert matplot to numpy array rgb and crop tight
        self.fig.canvas.draw()
        # print(self.fig.canvas.get_width_height())
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        # print(buf.shape)
        # ncols, nrows = self.fig.canvas.get_width_height()
        img = buf.reshape((self.ncols, self.nrows, 3)) # numpy, todo: need to crop it to reduce cpu overhead
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        return img

    def step(self, a):
        action = self._action_set[a]
        # self.ax.clear()
        # self.ax.axis('off')
        # print('action', action)
        reward = 0.0 # default if not purchased or sold during cycle
        # if isinstance(self.frameskip, int):
        # num_steps = self.frameskip
        # else:
            # num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(4): #(num_steps):
            self.Y = np.sin(self.X / self.freq) #/ self.amp
            # self.Y = np.sin(self.X /5) #/ self.amp
 
            # Valid sell
            if action == 2 and self.already_bought: 
                # self.already_bought = False
                self.done = True
                # reward += ( (self.Y - self.price) / self.price ) 
                reward = ( ( self.Y - self.price ) / 2  ) # zigmoid from -5 to 5
                # print (reward) 
                # print(' ', self.Y, self.price, self.Y - self.price)
                self.ax.plot(self.bar, self.Y,'b,')
                break
                
            # Valid buy
            elif action == 1 and not self.already_bought: 
                self.already_bought = True
                # reward = 0.0001    # incentive to buy?
                self.price = self.Y     # purchase price
                self.ax.plot(self.bar, self.Y, 'y,')
                self.ax.plot(62, -.95,'g,')   # bought indicator
            
            # invalid or no action
            else: 
                # reward -= 0.0001
                self.ax.plot(self.bar, self.Y,'w,') 
            
            # self.ax.plot(self.bar-1 , self.prevY,'k,') # erase previous pixel
            # self.prevY = self.Y
            # self.prevX = self.bar

            self.bar += 1
            self.X += 1

        if self.bar > 58:
            self.done = True # for simulating end of episode
            if reward == 0.0: # if self.already_bought : # end of chart, bought but did not sell
                reward = -0.9 # punish 
            # print('  end')
        
        ob = self._get_image()
        return ob, reward, self.done, {"bar": self.bar}


    def close(self):    # not sure if how used..
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        # self.ale.setInt(b'random_seed', seed2)
        # self.ale.loadROM(self.game_path)
        return [seed1, seed2]