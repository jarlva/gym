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
# import winsound
# import io
# from PIL import Image
from gym.envs.classic_control import rendering

class MyEnv(gym.Env, utils.EzPickle):
    metadata = {
      'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, frameskip=(2, 5), repeat_action_probability=0.):
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
        
        # setup plot      
        screen_height = 64 #200 #348 # 100 dots per inch
        screen_width = 64 #400 #678
        self.fig, self.ax = plt.subplots() 
        self.fig.set_size_inches(screen_width/100, screen_height/100)  # (w, h)
        # https://code.i-harness.com/en/q/8dd4b2
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.seed()
        self.viewer = None
        self.state = None
        self.viewer = rendering.SimpleImageViewer()

    def reset(self):
        # action logic
        self.bar = 2
        self.done = False
        # self.freq = random.randint(3,9)
        self.X = random.randint(1, 360)
        # self.amp = random.uniform(1.0,3.0)
        self.already_bought = False
        self.price = 0.0
        if self.render_counter > int(50000/ 6.2):  # control when to start rendering 6000 ~= 40,000 steps, every 6.7 steps
            self.ok_to_render = True
        self.render_counter += 1
        # plot reset
        self.ax.clear()
        # self.fig.patch.set_facecolor('black')
        self.ax.set_xlim(1, 64)
        self.ax.set_ylim(-1.05, 1.05)
        self.ax.axis('off')
        # print(' reset')
        # logger.info("reseting")
        return self._get_image()

    def render(self, mode='human'):
        # print ('', self.render_counter  )
        # img = self._get_image() # img is a numpy x,y,3 array, 210,160,3
        # if mode == 'rgb_array':
            # return img
        if mode == 'human': #and 
        # if self.ok_to_render: # render only after x steps
            # from gym.envs.classic_control import rendering
            # if self.viewer is None:
            # self.viewer = rendering.SimpleImageViewer()
            img = self._get_image() 
            self.viewer.imshow(img) # display the breakout inital screen
            return self.viewer.isopen

    def _get_image(self):
        # convert matplot to numpy array rgb and crop tight
        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        # ncols, nrows = self.fig.canvas.get_width_height()
        img = buf.reshape((64, 64, 3)) # numpy, todo: need to crop it to reduce cpu overhead
        return img

    def step(self, a):
        action = self._action_set[a]
        # print('action', action)
        reward = 0.0
        # if isinstance(self.frameskip, int):
        # num_steps = self.frameskip
        # else:
            # num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(4): #(num_steps):
            # self.Y = np.sin(self.X / self.freq) / self.amp
            self.Y = np.sin(self.X /5)

            # bought already
            # if action != 2 and self.already_bought: # already bought and not selling
            #     # if self.Y > self.price: 
            #         # reward += 0.01
            #     # else:
            #         # reward -= 0.01
            #     # reward = 0    
            #     self.ax.plot(self.bar, self.Y, 'k,')   # https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html
 
            # Valid sell
            if action == 2 and self.already_bought: 
                self.already_bought = False
                self.done = True
                # reward += ( (self.Y - self.price) / self.price ) 
                reward = ( self.Y - self.price ) / 2
                if reward > 0.8: # discoredge small gains in favor of big(?)
                    reward = 1.0 
                # if (self.Y - self.price) > 0.4:
                    # reward = 1.0
                # else:
                    # reward -= 1.0
                # print(' ', self.Y, self.price, self.Y - self.price)
                # if reward > 0:
                    # print ('  buy ', self.price, ' sell', self.Y, 'reward', reward)
                self.price = 0.0
                self.ax.plot(self.bar, self.Y,'b,')
                break
                
            # Valid buy
            elif action == 1 and not self.already_bought: 
                self.already_bought = True
                # reward += 0.0001    # incentive to buy?
                self.price = self.Y     # purchase price
                self.ax.plot(self.bar, self.Y, 'r,')

            # Invalid buy
            # elif action == 1 and self.already_bought: #already bought and agent wants to buy again
            #     reward -= 0.5   # punish
            #     # self.done = True
            #     # print(' punish', reward)
            #     self.ax.plot(self.bar, self.Y,'r,')
            #     # break
 
            # # Invalid sell
            # elif action == 2 and not self.already_bought: 
            #     reward -= 0.5   # punish
            #     # self.done = True
            #     # print(' punish', reward)
            #     self.ax.plot(self.bar, self.Y,'r,')
            #     # break

            # invalid or no action
            else: 
                # reward -= 0.0001
                self.ax.plot(self.bar, self.Y,'k,') 
            
            self.bar += 1
            self.X += 1

        if self.already_bought: self.ax.plot(63, -.95,'g,') # bought indicator

        if self.bar > 58:
            self.done = True # for simulating end of episode
            # bought but did not sell and reached end of chart
            if self.already_bought: # end of chart
                reward = -1 # punish 
            # print('  end')
        
        # if reward > 1.9: print(' ', reward)
        ob = self._get_image()
        return ob, reward, self.done, {"bar": self.bar}

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