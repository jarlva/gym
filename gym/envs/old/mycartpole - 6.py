''' https://github.com/jinfagang/rl-solution/blob/master/solve_cart_pole.py
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
https://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
PowerTransformer 	2097 episode [01:21	1.06
faster      		2784 episode [01:54
					2296 episode [01:24
					
RobustScaler	 	1853 episode [01:16	1.95
					2560 episode [01:35
					3775 episode [02:21
'''
import gym
from gym import spaces #, logger
from gym.utils import seeding
import numpy as np
import random
from sklearn.preprocessing import RobustScaler, PowerTransformer, StandardScaler, QuantileTransformer, Normalizer  # minmax_scale #,  #, 

import time
from pathlib import Path
import shutil

class MyCartPoleEnv(gym.Env):
    metadata = { 'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50 }

    def __init__(self):

        sine =  True
        if sine: file = '2_Sine_noise.csv'
        else: file = 'EWZ_30_Min.csv' #'XBI_30_Min.csv'
        r = open('setting.txt', "r") # save csv file in test dir
        dirloc = Path(r.readline())
        r.close()
        shutil.copy2(str(file), str(dirloc))
        read = np.loadtxt(file, skiprows=8 , delimiter=' ', dtype=np.float64 ) # skip first 7 rows for heading and intia bad data        
        # read -= read.min() # remove dc level and add small num to avoid devide by zero
        # read[:,1] -= read[:,1].min()
        # print( 'Min:{0:5.2f}, Max:{1:5.2f}'.format(read[:,3].min() , read[:,3].max() ) )
        # read = minmax_scale(read, feature_range=(0, 0.999), copy= False) # Normalize data -1,1. https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
        # scaler = QuantileTransformer(copy=False) #sine:2.9 not good 
        # scaler = Normalizer(copy=False) #sine:6 not good
        # scaler = StandardScaler(copy=False) # sine:1.2 1849,01:19 3614,04:10 1569,01:31
        # scaler = PowerTransformer(copy=False) #sine:1.06 2350,02:18 1849,01:23 2865,02:49
        scaler = RobustScaler(copy=False) #sine:2.2        4312,04:19 1403,00:54
        read = scaler.fit_transform(read)

        if sine:    # For sinewave
            self.data = read[:,0]
            self.g1 = read[:,1]
            self.mul = 2.2 #1.2 #2.2
            # read = read* 2.2
        else:   # 0:pkmd 1:pkwd 2:cxd 3:cxmd 4:Close 5:pkw 6:pkm
            self.data = read[:,2]
            self.g1 = read[:,1]
            self.g2 = read[:,1]
            self.g3 = read[:,0]
            #RobustScaler XBI 0:pkmd  1:0.12  2:0.17  3:3.3  4:0.84  5:pkw  6:? Sine: 2.2
            #RobustScaler EWZ 0:pkmd  1:0.12  2:0.08  3:3.3  4:0.84  5:pkw  6:? Sine: 2.2
            self.mul = 0.08


        self.scale = (self.data.max() - self.data.min()) 
        print(file, 'Scale:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.scale, self.data.min(), self.data.max()) )
        print('g1:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.g1.max()-self.g1.min(), self.g1.min(), self.g1.max()) )

        # how many bars per episode, increasing from 170 to 255 increased score rate of increase.
        # increasing it more (> 300) causes score to drop permanetly
        self.cycle = 256
        # 252 how many previous bars to include. increasing number increased score rate of increase.
        self.prevbars = 252
        self.prevbars1 = 64
        self.prevbars2 = 64
        self.prevbars3 = 32
        self.maxseries = max(self.prevbars, self.prevbars1, self.prevbars2, self.prevbars3)
        
        low = np.concatenate((
            np.full(self.prevbars, self.data.min()),      # data/price
            # np.full(self.prevbars1, self.g1.min()),       # g1
            #np.full((self.prevbars2), self.g2.min()),     # g2
            #np.full((self.prevbars3), self.g3.min()),     # g3
            [0,                                             # bar
            self.data.min()]))                              # buy price
        high = np.concatenate((                             # close
            np.full(self.prevbars, self.data.max()),      # data/price
            # np.full(self.prevbars1, self.g1.max()),       # g1 
            #np.full((self.prevbars2), self.g2.max()),     # g2
            #np.full((self.prevbars3), self.g3.max()),     # g3
            [self.cycle-1,                                  # bar
            self.data.max()]))                              # buy price

        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.action_space = spaces.Discrete(3)  # noop/buy/sell
        self.reward_range = (-1.0, 1.0)
        self.viewer = None
        self.state = None
        self.okrender = False
        self.ok_to_render = False # if okrender true will be activated after X steps
        self.stepcount = 0
        self.bar = 0
        self.maximum, self.minimum = 0.0, 0.0
        # setup plot      
        self.screen_height=200 #200 #348 # 100 dots per inch
        self.screen_width= self.cycle-2 #400 #64 #400 #678
        self.seed()
        # self.reset() already done in main.py

    def _get_ob(self):
        data = self.data[self.X - self.prevbars+1:self.X+1]
        state = np.concatenate(( data, (self.bar, self.price))) # for some reason needs extra ()
        # g1 = self.g1[self.X - self.prevbars1+1:self.X+1]
        # state = np.concatenate((data, g1, (self.bar, self.price))) # for some reason needs extra ()
        #g2 = self.g2[self.X - self.prevbars2+1:self.X+1] 
        #state = np.concatenate(( data, g1, g2, (self.bar, self.price))) # for some reason needs extra ()
        #g3 = self.g3[self.X - self.prevbars3+1:self.X+1]
        #state = np.concatenate(( data, g1, g2, g3, (self.bar, self.price)))
        return state #, dtype=np.float32)

    def step(self, action):     # **********************************************
        reward = 0.0
        self.prev = self.Y
        self.X += 1
        self.bar += 1
        self.Y = self.data[self.X]
        #cxd = self.g1[0]
        # cxd = self.g1[self.X]

        # reached end
        if self.bar == self.cycle-1:
            self.done = True # for simulating end of episode
            reward = -1 #- self.price  # punish. otherwise it will keep going to bar limit with 0 reward.. 

        # Buy
        elif action == 1 and not self.already_bought: 
            self.already_bought = True
            # if self.ok_to_render: self.ax.plot(self.bar, self.Y,'r,') 

            self.price = self.Y     # purchase price
            #self.price = (1 - (self.Y - self.data.min()) / self.scale )  - 0.5 
            #reward = -self.Y    
 
            #if reward > self.minimum:   # record max reward
            #    self.minimum = reward
            #    print(' nBuy:{0:4.2f}'.format(reward))
            #reward = np.clip( reward , -0.999, 0.999)

        # Sell
        elif action == 2 and self.already_bought:
            self.done = True
            # if self.ok_to_render: self.ax.plot(self.bar, self.Y, 'b.') 
            
            reward = (self.Y - self.price) * self.mul
            #reward = self.Y
            #reward = (( (self.Y - self.data.min()) / self.scale  -0.5) + self.price) * 4.5  

            if reward > self.maximum:   # record max reward
                self.maximum = reward
                print(' nSell:{0:5.2f}'.format(self.maximum))

            reward = np.clip( reward , -0.999, 0.999)
            # reward = np.clip( ( self.Y - self.price ) , -0.99, 0.99)    # sine -1,1

        #bought but no action
        # elif self.already_bought and self.Y < self.price:
        #    reward = -0.004

        # remarked for training speed, enable for rendering            
        # if self.ok_to_render: 
        #     if self.already_bought: self.ax.plot(self.bar, self.Y,'y,') 
        #     else: self.ax.plot(self.bar, self.Y,'w,')

        return self._get_ob(), reward, self.done, {"bar": self.bar} # , dtype=np.float32

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

    def reset(self):
        self.stepcount += self.bar
        self.bar = 0
        self.done = False
        self.already_bought = False
        self.price = 0.0
        self.X = random.randint(self.maxseries, int(self.data.size-self.cycle-1))
        self.prev = self.data[self.X-1]
        self.Y = self.data[self.X]
        # plot reset
        if self.ok_to_render:
        # if self.okrender and self.stepcount < 200000:  #  1000 ~= 4,500 steps, every 2.1  steps
            self.ax.clear()
            self.ax.set_xlim(0, self.screen_width )
            # self.ax.set_ylim(-1.01, 1.01)
            self.ax.set_ylim(self.data.min(), self.data.max())
            self.ax.axis('off')
            time.sleep(2) # give a chance to review 
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

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]