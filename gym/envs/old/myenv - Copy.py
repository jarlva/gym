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
# import io
# from PIL import Image

class MyEnv(gym.Env, utils.EzPickle):
    metadata = {
      'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, frameskip=(2, 5), repeat_action_probability=0.):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(self, frameskip, repeat_action_probability)
        self._observation = []
        self.frameskip = frameskip
        self.done = False
        self._action_set = np.array( [0, 1, 2], dtype=int )  # noop/buy/sell
        self.action_space = spaces.Discrete(len(self._action_set))
        
        # setup plot      
        screen_height = 160 #200 #348
        screen_width = 210 #400 #678
        self.fig, self.ax = plt.subplots() 
        self.fig.set_size_inches(2.1,1.6)
        # https://code.i-harness.com/en/q/8dd4b2
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.seed()
        self.viewer = None
        self.state = None

    def reset(self):
        # plot reset
        self.ax.clear()
        # self.fig.patch.set_facecolor('black')
        self.ax.set_xlim(0, 360)
        self.ax.set_ylim(-1.0, 1.0)
        # self.ax.axis('off')
        print(' reset')
        # logger.info('reset')

        self.bar = 0
        self.done = False
        # self.ale.reset_game()
        # logger.info("reseting")
        return self._get_obs()

    def render(self, mode='human'):
        img = self._get_image() # img is a numpy x,y,3 array, 210,160,3
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img) # display the breakout inital screen
            return self.viewer.isopen

    def _get_image(self):
        # convert matplot to numpy array rgb and crop tight
        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        ncols, nrows = self.fig.canvas.get_width_height()
        img = buf.reshape((nrows, ncols, 3)) # numpy, todo: need to crop it to reduce cpu overhead
        return img

    def step(self, a):
        reward = 0.0
        action = self._action_set[a]
        # print('action', action)
        if self.bar  > 20 : # at end of chart 
            self.done = True # for simulating end of episode
            print('end')
        
        # simulate a sine wave and draw on as matrix pixles
        # self.X = np.arange(0.0,  self.bar/10, 0.1)
        # self.Y = np.sin(self.X*1)
        # self.ax.plot(self.X, self.Y, scalex=False, color="white" )        
        # self.fig.canvas.draw()
        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            self.X = self.bar
            self.Y = np.sin(self.X/10)
            self.ax.plot(self.X, self.Y,'k.')
            self.bar += 1
            reward += 1 #self.ale.act(action)

        # print(self.bar)
        ob = self._get_obs()
        return ob, reward, self.done , {"bar": self.bar}

    def _get_obs(self):
        img = self._get_image()
        return img

    def close(self):    # not sure if how used..
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]