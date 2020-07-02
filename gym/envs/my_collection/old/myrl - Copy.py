# https://github.com/jinfagang/rl-solution/blob/master/solve_cart_pole.py
import gym, random, time, shutil, os, socket #, datetime
from gym import spaces #, logger
from gym.utils import seeding
import numpy as np
from sklearn.preprocessing import QuantileTransformer, RobustScaler #, MinMaxScaler, minmax_scale #, PowerTransformer, StandardScaler, , Normalizer  # minmax_scale #,  #, 
from pathlib import Path
import win32file, win32con # for file attributes

# for REST
# from flask import Flask
# from flask_restful import Api, Resource, reqparse
# app = Flask(__name__)
# api = Api(app)
# api.add_resource(User, "/")
# app.run(debug=True)

if os.name == 'nt': nt =True
else: nt = False
RT =  True

class MyrlEnv(gym.Env): # for RL
# class MyCartPoleEnv(gym.Env): 
    metadata = { 'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50 }

    def __init__(self):

        sine =  True
        if sine: filen = '2_Sine_noise.csv' #'1_Sine.csv' '2_Sine_noise.csv'
        else:    filen ='EWZ_3_Min.csv'
        
        if nt: 
            pathfile = Path(r'C:\Users\Jake\py\Deep-RL-Keras')
        elif not nt and (socket.gethostname() == "BONGO" or socket.gethostname() == "Meshy"):
            pathfile = Path(r'/mnt/c/Users/Jake/py/Deep-RL-Keras')
        else: 
            pathfile = Path(r'/root/Deep-RL-Keras')

        self.data_file = (r'E:\data.csv')
        self.orders_file = (r'E:\order.csv')
        # print('================== path:', pathfile )
        file = str(pathfile / filen)
        if not RT:
            r = open('setting.txt', "r") # save csv file in test dirrr
            dirloc = Path(r.readline())
            r.close()
            shutil.copy2(str(file), str(dirloc))

        self.read = np.loadtxt(file, skiprows=8 , delimiter=' ', dtype=np.float64 ) # skip first 7 rows for heading and intia bad data        
        # self.read -= self.read.min() # remove dc level and add small num to avoid devide by zero
        # read[:,1] -= read[:,1].min()
        # print( 'Min:{0:5.2f}, Max:{1:5.2f}'.format(read[:,3].min() , read[:,3].max() ) )
        # read = minmax_scale(read, feature_range=(0, 0.999), copy= False) # Normalize data -1,1. https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
        # self.minmax = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = QuantileTransformer(output_distribution='uniform') #copy=False, #sine:2.9 not good 
        qt = self.scaler.fit_transform(self.read) -0.5 #* 2 - 1
        self.scaler1 = RobustScaler() # copy=False) #sine:2.2
        rs = self.scaler1.fit_transform(self.read)

        if sine: # benchmark settings for sine
            self.mul = 2 #0.5 #2.17 #5.4 #5.7 #5.4 #2.17
            eps = 64
            self.cycle = int(1.5 * eps)#; print(self.cycle)  #65 is about 1 cycle. performence was the same even at 80
            # self.cycle = 200 # should be identical to _init__ max_episode_steps

            # 0:data 1:pkwd 2:pkmd 3:pkw 4:pkm
            self.datacol = 0 ; self.data = rs[:,self.datacol] #rs[:,0] #* 0.63 #* 2.17
            self.prevbars = 1 #int(2 * eps) #1 #int(2 * eps) #eps

            self.g1col = 1 ; self.g1 = qt[:,self.g1col]
            self.prevbars1 = int(2 * eps) #210 #252

            self.g2col = 2
            self.g2 = qt[:,self.g2col]
            self.prevbars2 = 0 #int(eps / 2)  #252

        else:
            self.mul = 7 #2.1 #5min:7
            eps = 110
            self.cycle = int(1.5 * eps) #120

            # 0:pkmd 1:pkwd 2:cxd 3:cxmd 4:Close 5:pkw 6:pkm 7:cx 8:rsi
            self.data = qt[:,3] #* 2
            self.prevbars = 1 #int(eps/2)
            
            self.g1 = qt[:,0] # 0
            self.prevbars1 = 2 * eps #250
            
            self.g2 = qt[:,8]
            self.prevbars2 = 2 * eps #0 #int(eps/2)
            
            # self.g3 = read[:,2]
            # self.prevbars3 = 0

        self.maxseries = max(self.prevbars, self.prevbars1, self.prevbars2) #, self.prevbars3)
        self.X = self.maxseries - 1    # for RT, constant
        
        self.scale = (self.data.max() - self.data.min()) 
        print(file)#, 'Scale:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.scale, self.data.min(), self.data.max()) )
        print('data Range:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.data.max()-self.data.min(), self.data.min(), self.data.max()) )
        if self.prevbars1 > 0: print('  g1:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.g1.max()-self.g1.min(), self.g1.min(), self.g1.max()) )
        if self.prevbars2 > 0: print('  g2:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.g2.max()-self.g2.min(), self.g2.min(), self.g2.max()) )
        # exit()
        
        low = np.hstack((
            np.full(self.prevbars, self.data.min()),      # data/price  self.data.min()),
            np.full(self.prevbars1, self.g1.min()),     # g1
            # np.full((self.prevbars2), self.g2.min()),   # g2
            #np.full((self.prevbars3), self.g3.min()),    # g3
            # 0,                                          # bar
            -self.data.max()-self.data.min()              # buy price
            ))             
        high = np.hstack((                             
            np.full(self.prevbars, self.data.max()),      # data/price
            np.full(self.prevbars1, self.g1.max()),     # g1 
            # np.full((self.prevbars2), self.g2.max()),   # g2
            #np.full((self.prevbars3), self.g3.max()),    # g3
            # self.cycle,                                 # bar
            self.data.max()-self.data.min()               # buy price
            ))

        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.action_space = spaces.Discrete(3)  # noop/buy/sell
        self.reward_range = (-1.0, 1.0)
        self.viewer = None
        self.state = None
        self.okrender = False
        self.ok_to_render = False # if okrender true will be activated after X steps
        self.bar, self.stepcount = 0, 0
        self.maximum, self.minimum = 0.0, 0.0
        # setup plot      
        self.screen_height=200 #200 #348 # 100 dots per inch
        self.screen_width= self.cycle-2 #400 #64 #400 #678
        self.seed()
        self.old = os.path.getmtime(self.data_file)

    def read_data(self):
        time.sleep(0.05)
        while True: # Wait for new data bar from WL
            fattrs = win32file.GetFileAttributes(self.data_file)
            if fattrs & win32con.FILE_ATTRIBUTE_ARCHIVE:
                win32file.SetFileAttributes(self.data_file, 128) # reset archive bit
                break
        self.read = np.loadtxt(self.data_file, skiprows=0 , delimiter=' ', dtype=np.float64 ) # skip 
        rs = self.scaler1.fit_transform(self.read)
        qt = self.scaler.fit_transform(self.read) -0.5
        self.data = rs[:,self.datacol]
        self.g1 = qt[:,self.g1col]
        return

    def _get_ob(self):
        data = self.data[self.X - self.prevbars+1:self.X+1]
        # state = np.hstack(( data, self.bar, self.price)) #bar not used anymore
        # state = np.hstack(( data,  self.price))
        g1 = self.g1[self.X - self.prevbars1+1:self.X+1]
        # state = np.hstack((data, g1, self.bar, self.price))
        state = np.hstack((data, g1, self.price))
        # assert state.size==133, str (state.size + " " + data.size  + " " + g1.size  + " " + self.bar + " " + self.price)
        # if state.size < 133:
        #     print('\r\nstate:', state.size)
        #     print('data:', data.size)
        #     print('g1:', g1.size)
        #     print('bar,price:',self.bar, self.price) 
        #     print('self.X', self.X)
        #     exit()
        # state = np.hstack((data, g1, (0, self.price)))
        # g2 = self.g2[self.X - self.prevbars2+1:self.X+1] 
        # state = np.hstack(( data, g1, g2, (self.bar, self.price))) # for some reason needs extra ()
        #g3 = self.g3[self.X - self.prevbars3+1:self.X+1]
        #state = np.hstack(( data, g1, g2, g3, (self.bar, self.price)))
        return state #, dtype=np.float32)

    def step(self, action):     # **********************************************
        reward = 0.0
        self.bar += 1
        if not RT: self.X += 1
        else: 
            self.read_data() # wait for data file to update 
        # if RT:        
            # Update orders file 
            f = open(self.orders_file, "w")
            f.writelines(str(action))    # save elapsed time/episodes
            f.close()
            # time.sleep(0.02) # gaurd time 
        
        # reached end
        # if self.bar == self.cycle-1:
        #     self.done = True # for simulating end of episode
        #     reward = -1 #- self.price  # punish. otherwise it will keep going to bar limit with 0 reward.. 

        # Buy
        if not self.already_bought and action == 1: 
            self.already_bought = True
            Y = self.data[self.X]
            self.price = Y     # purchase price
            #self.price = (1 - (self.Y - self.data.min()) / self.scale )  - 0.5 
            if self.ok_to_render: self.ax.plot(self.bar, Y, 'g,')

        # Sell
        elif self.already_bought and action == 2:
            Y = self.data[self.X]
            reward = (Y - self.price) * self.mul
            if reward > self.maximum:   # record max reward
                self.maximum = reward
                print(' nHigh:{0:5.2f}'.format(self.maximum))
            # elif reward < self.minimum:   # record max reward
            #    self.minimum = reward
            #    print(' nLow:{0:4.2f}'.format(reward))
            # reward = np.clip( reward , -1.0, 1.0)

            if self.ok_to_render: self.ax.plot(self.bar, Y, 'r,') 
                # wait = input("PRESS ENTER TO CONTINUE.")
                # time.sleep(3)
                # print(' nSell:{0:5.2f}'.format(reward))
            self.done = True

        # remarked for training speed, enable for rendering            
        elif self.ok_to_render: 
            Y = self.data[self.X]
            if self.already_bought: self.ax.plot(self.bar, Y, 'b,') 
            else: self.ax.plot(self.bar, Y, 'w,')

        return self._get_ob(), reward, self.done, {"bar": self.bar} # , dtype=np.float32

    def reset(self):
        # print('bar:', self.bar)
        # self.stepcount += self.bar
        self.bar = 0
        self.done, self.already_bought = False, False
        self.price = 0.0
        # if not self.ok_to_render: # training
            # self.X = self.prevbars-1
        if not RT: self.X = random.randint(self.maxseries-1, int(self.data.size-self.cycle))
        # plot reset
        # if self.okrender and self.stepcount < 200000:  #  1000 ~= 4
        if self.ok_to_render:
            # time.sleep(3) # give a chance to review 
            self.ax.clear()
            self.ax.set_xlim(0, self.screen_width )
            self.ax.set_ylim(self.data.min(), self.data.max())
            self.ax.axis('off')
            # wait = input("PRESS ENTER TO CONTINUE.")

        # self.X = random.randint(200, int(self.read[:,0].size-self.cycle-1))
        # self.scaler1 = RobustScaler() # copy=False) #sine:2.2
        # rs = self.scaler1.fit_transform(self.read)
        # self.scaler = QuantileTransformer(output_distribution='uniform')
        # a = self.read[self.X - 199:self.X + self.cycle + 1,]    # slice only the window from read
        # qt = self.scaler1.fit_transform(a)                      # then normalize it
        # # qt = minmax_scale(qt, feature_range=(-1.0, 1.0))
        # qt = self.minmax.fit_transform(qt)
        # self.data = qt[:,0]   #* 2 - 1
        # self.X = 199

        return self._get_ob()

    def render(self, mode='human'):
        # return# self.viewer.isopen
        # if self.ok_to_render: # render only after x steps
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            import matplotlib.pyplot as plt
            # time.sleep(2.3)
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
