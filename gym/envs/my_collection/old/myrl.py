# https://github.com/jinfagang/rl-solution/blob/master/solve_cart_pole.py
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
import gym, random, shutil, os, socket, time#, datetime
from gym import spaces #, logger
from gym.utils import seeding
import numpy as np
from sklearn.preprocessing import QuantileTransformer , RobustScaler #MinMaxScaler #, StandardScaler #, , , PowerTransformer  #, Normalizer
from pathlib import Path

# from gym.envs.classic_control import rendering
# import matplotlib.pyplot as plt

# for REST
# from flask import Flask
# from flask_restful import Api, Resource, reqparse
# app = Flask(__name__)
# api = Api(app)
# api.add_resource(User, "/")
# app.run(debug=True)

if os.name == 'nt': 
    nt =True
    import win32file, win32con # for file attributes
else: nt = False

RT = not True
# RT = True

class MyrlEnv(gym.Env): # for RL
# class MyCartPoleEnv(gym.Env): 
    metadata = { 'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50 }
    def __init__(self):

        sine = not True
        sine = True
        if sine: filen = '2_Sine_noise_large.csv' # '1_Sine.csv' _rand '1_Sine.csv' '2_Sine_noise.csv' '2_Sine_noise_small.csv' 2_Sine_noise_large
        else:    filen = 'XBI_130.csv' #'XBI_Daily.csv' #'EWZ_130.csv'  'EWZ_30_Min.csv' 'EWZ_130.csv'
        
        if nt: 
            pathfile = Path(r'C:\Users\Jake\py\Deep-RL-Keras')
        elif not nt and (socket.gethostname() == "BONGO" or socket.gethostname() == "Meshy"):
            pathfile = Path(r'/mnt/c/Users/Jake/py/Deep-RL-Keras')
        else:
            pathfile = Path(r'/root/Deep-RL-Keras')

        self.data_file = (r'R:\data.csv')
        self.orders_file = (r'R:\order.csv')
        # print('================== path:', pathfile )
        file = str(pathfile / filen)
        if not RT:
            r = open('setting.txt', "r") # save csv file in test dir
            dirloc = Path(r.readline())
            r.close()
            shutil.copy2(str(file), str(dirloc))
            print(str(dirloc))

        self.read = np.loadtxt(file, skiprows=1 , delimiter=' ', dtype=np.float64 ) # skip first 7 rows for heading and intia bad data        
        # self.read -= self.read.min() # remove dc level and add small num to avoid devide by zero
        # read[:,1] -= read[:,1].min()
        # print( 'Min:{0:5.2f}, Max:{1:5.2f}'.format(read[:,3].min() , read[:,3].max() ) )
        # read = minmax_scale(read, feature_range=(0, 0.999), copy= False) # Normalize data -1,1. https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
        # self.minmax = MinMaxScaler(feature_range=(-1, 1)).fit(self.read)
        # minmax = self.minmax.transform(self.read)
        # self.scaler1 = StandardScaler()
        # self.scaler1 = PowerTransformer()
        # self.scaler1 = RobustScaler() ; 
        # rs = RobustScaler().fit_transform(self.read)
        qt = QuantileTransformer(output_distribution='uniform', random_state=1).fit_transform(self.read) * 2 - 1 # important to make qt -/+, not 0/1 #copy=False, #sine:2.9 not good 
        # qt = self.scaler.fit_transform(self.read) * 2 - 1 # important to make qt -/+, not 0/1
        if RT: self.QTscaler = QuantileTransformer(output_distribution='uniform', random_state=1).fit(self.read)
        # if RT: self.scaler = QuantileTransformer(output_distribution='uniform', random_state=1).fit(self.read)

        if sine: # benchmark settings for sine
            # raw:0.25, rs/qt:100, StandardScaler:60, minmax:104
            # Windowed normalization: data/qt: 530, minmax:
            # self.mul = 1  #00 #104 #104 #645 #108
            eps = 64
            self.cycle = 120 #int(1.5 * eps)#; print(self.cycle)  #65 is about 1 cycle. performence was the same even at 80

            # 0:data 1:pkwd 2:pkmd 3:pkw 4:pkm
            self.datacol = 1 ; self.data = qt[:,self.datacol]
            self.prevbars = 24 # int(1.4 * eps) #1 #int(2 * eps) # pure sine: 24

            self.g1col = 2 ; self.g1 = qt[:,self.g1col]
            self.prevbars1 = 2 #int(1.3 * eps) #10 #40 #int(1.4 * eps)

            self.g2col = 2 ; self.g2 = qt[:,self.g2col]
            self.prevbars2 = 10
        else:
            # self.mul = 10 #4.4
            eps =  64 #110
            self.cycle = 180 #130 * 2 #130 for daily, for 30 min: 130 * (390/30)

            # 0:Close 1:cxmd 2:cxd 3:pkwd  H33
            # 0:Close 1:cxmd 2:cxd 3:cx 4:pkwd
            self.datacol = 2 ; self.data = qt[:,self.datacol]
            self.prevbars = 24  #* 2 #13 * 24 #24. 24 for EWZ daily cxd
            
            self.g1col = 3 ; self.g1 = qt[:,self.g1col]
            self.prevbars1 = 6 #int(1.5 * eps)
            
            self.g2col = 1 ; self.g2 = qt[:,self.g2col]
            self.prevbars2 = 10 #int(2 * eps)

        self.maxseries = max(self.cycle, self.prevbars)  #2 * self.cycle #1 * max(self.prevbars, self.prevbars1, self.prevbars2) #, self.prevbars3)
        self.X = self.maxseries - 1  # for RT, constant. can be left always on
        
        self.scale = (self.data.max() - self.data.min()) 
        print(file)#, 'Scale:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.scale, self.data.min(), self.data.max()) )
        print('Read Range:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.read[ : , self.datacol].max()-self.read[ : , self.datacol].min(), self.read[ : , self.datacol].min(), self.read[ : , self.datacol].max()) )
        print('data Range:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.data.max()-self.data.min(), self.data.min(), self.data.max()) )
        if self.prevbars1 > 0: print('  g1:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.g1.max()-self.g1.min(), self.g1.min(), self.g1.max()) )
        if self.prevbars2 > 0: print('  g2:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.g2.max()-self.g2.min(), self.g2.min(), self.g2.max()) )
        
        low = np.hstack((
            np.full(self.prevbars, self.data.min()),     # data/price  self.data.min()),
            # np.full(self.prevbars1, self.g1.min()),      # g1
            # np.full((self.prevbars2), self.g2.min()),  # g2
            0,                                           # self.already_bought
            # self.data.min() - self.data.max(),           # self.delta
            # 0,                                         # self.bar
            # 0, # self.data.min()                       # bar buy price
            # self.data.min()                            # buy price
            ))             
        high = np.hstack((                             
            np.full(self.prevbars, self.data.max()),     # data/price
            # np.full(self.prevbars1, self.g1.max()),      # g1 
            # np.full((self.prevbars2), self.g2.max()),  # g2
            1,                                           # self.already_bought
            # self.data.max() - self.data.min(),           # self.delta
            # self.cycle  #self.data.max(                # bar buy price
            # self.data.max()                            # buy price
            ))

        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.action_space = spaces.Discrete(3)  # noop/buy/sell
        self.reward_range = (-2.0, 2.0)
        # self.reward_range = (self.data.min()-self.data.max(), self.data.max()-self.data.min())
        self.viewer, self.state = None, None
        self.okrender = False
        self.ok_to_render = False # if okrender true will be activated after X steps
        self.bar, self.stepcount = 0, 0
        self.maximum, self.minimum, self.delta = 0.0, 0.0, 0.0
        # setup plot      
        self.screen_height=200 #200 # 100 dots per inch
        self.screen_width= self.cycle   #400 #64 #400 #678
        self.seed()
        #self.old = os.path.getmtime(self.data_file)

    def _get_ob(self):
        # print(self.X)
        # Best: self.already_bought, self.delta: 2.21m
        # Adding self.bar: not good
        # removing self.already_bought and leaving self.delta: slows down a bit. 2:21m
        # self.already_bought, self.delta, self.cycle-self.bar:not good
        data = self.data[self.X - self.prevbars+1:self.X+1]
        state = np.hstack(( data, self.already_bought)) #, self.delta)) #, self.price))
        # g1 = self.g1[self.X - self.prevbars1+1:self.X+1]
        # state = np.hstack((data, g1, self.already_bought)) #, self.delta)) #, self.price))
        # g2 = self.g2[self.X - self.prevbars2+1:self.X+1] 
        # state = np.hstack(( data, g1, g2, self.stepcount, self.price))
        #g3 = self.g3[self.X - self.prevbars3+1:self.X+1]
        #state = np.hstack(( data, g1, g2, g3, (self.bar, self.price)))
        # if self.bar > 45: print('self.bar ' + str(self.bar))
        # assert state.size==self.observation_space.high.size, str(state.size) + " " \
        #     + str(data.size) + " " + str(g1.size) + " " + str(self.X) + " " + str(self.bar)
        return state #, dtype=np.float32)
    
    def read_data(self):
        # time.sleep(0.15)
        while True: # Wait for new data bar from WL
            time.sleep(0.07)
            fattrs = win32file.GetFileAttributes(self.data_file)
            if fattrs & win32con.FILE_ATTRIBUTE_ARCHIVE:
                win32file.SetFileAttributes(self.data_file, 128) # reset archive bit
                break
        self.read = np.loadtxt(self.data_file, skiprows=0 , delimiter=' ', dtype=np.float64) # skip 
        # rs = self.scaler1.fit_transform(self.read)
        qt = self.QTscaler.transform(self.read) # preserve true price, only for the data column
        # minmax = self.minmax.transform(self.read)
        # self.data = minmax[ -self.maxseries: , self.datacol]
        # assert qt.__len__() == self.cycle, 'data.csv length must be: ' + self.cycle
        # Normalize only L/T indicators, like cxd, cxmd. Has to match reset normalization
        self.data = qt[ : , self.datacol] * 2 - 1 # -self.maxseries
        self.g1 = qt[ : , self.g1col] * 2 - 1
        # self.g2 = self.read[ -self.maxseries:, self.g2col]
        # qt = self.scaler.transform(self.read)  * 2 - 1
        # self.g1 = qt[ -self.maxseries: , self.g1col]  * 2 - 1
        # self.g2 = qt[ -self.maxseries: , self.g2col]  * 2 - 1
        return

    def reset(self):
        # print('bar:', self.bar)
        # print(self.data[-self.cycle:].min(), self.data[-self.cycle:].max())
        # print(self.g1[-self.cycle:].min(), self.g1[-self.cycle:].max())
        # self.stepcount = 0,
        # self.delta = 0.0
        self.bar = 0
        self.price = 0.0
        self.done, self.already_bought = False, False

        self.X = random.randint( self.maxseries - 1, self.read.__len__() - self.cycle -1) # TS remarked for training speed
        # if not RT: self.X = random.randint( self.maxseries - 1, self.read.__len__() - self.cycle -1)
            # self.X = 2054 ; print(self.X) # for debug


        # Normalize window, use only for L/T indicators
        # a = self.read[self.X - self.maxseries + 1 : self.X + self.cycle + 1, ]    # slice only the window from read
        # rs = self.scaler1.fit_transform(a)
        # qt1 = self.QTscaler.transform(a) # preserve true price, only for the data column
        # minmax = self.minmax.transform(a)
        # self.data = minmax[ : , self.datacol]
        # self.data = a[ : , self.datacol] # no normalization, preserve true price
        # qt = self.scaler.fit_transform(a)
        # self.data = qt[ : , self.datacol] * 2 - 1
        # assert self.data.min() > 0.09
        # self.g1 = qt1[ : , self.g1col]  * 2 - 1 # normalize only cxd
        # self.g2 = qt1[ : , self.g2col]  * 2 - 1 # no normalization, preserve true price
        # self.X = self.maxseries - 1 # -1 beacuse step adds 1

        # print (self.data.min(), self.data.max())
        # print (self.g1.min(), self.g1.max())
        # assert self.X > self.cycle and self.X < (self.data.size-self.cycle - 2), \
        #     str(self.cycle) + " " + str(self.data.size)  \
        #     + " " + str(self.maxseries)  + " " + str(self.X)
        # plot reset
        # if self.okrender and self.stepcount < 200000:  #  1000 ~= 4

        # if self.ok_to_render:
        #     self.ax.clear()
        #     self.ax.set_xlim(0, self.screen_width + 1)
        #     self.ax.axis('off')

            # self.ax.set_ylim(-1.04, 1.04)
            # self.ax.set_ylim(-2, 2)
        
        #     time.sleep(2) # give a chance to review 
        #     # wait = input("PRESS ENTER TO CONTINUE.")
        return self._get_ob()
    
    def step(self, action):     # **********************************************
        reward = 0.0 ; self.bar += 1
        
        self.X += 1 # active only in training for speedup
        # TS remarked for training speed
        # if not RT: self.X += 1 # remark for training speed
        # else: 
        #     self.read_data() # wait for data file to update
        #     # if self.ok_to_render: self.ax.plot(self.bar, 0, 'c,') # for debug
            
        # # reached end
        if self.bar == self.cycle:
            self.done = True # end of episode
            reward = -2 # punish 
            # if RT and self.already_bought: self.Update_order(2) # if bought, tell WL to sell(?)

        # invalid action. 
        # Notes: excluding this condition reduced/improved both max and mean bars 
        # while finishing at the same time as with it. So, don't use it.
        # max bars used with just cxd was 140 while self.cycle was 200.
        # elif (not self.already_bought and action == 2) or (self.already_bought and action == 1):
        #     self.done = True
        #     # reward = -0.01
            # if self.ok_to_render: print('invalid', action, self.bar)

        # Buy         
        elif not self.already_bought and action == 1: 
            self.already_bought = True
            self.price = self.data[self.X]
            #self.price = (1 - (self.Y - self.data.min()) / self.scale )  - 0.5 
         
            # if self.ok_to_render: self.ax.plot(self.bar, self.price, 'w,')
            # if RT: self.Update_order(action)
                
        # Sell
        elif self.already_bought and action == 2: 
            self.done = True
            Y = self.data[self.X]
            reward = (Y - self.price) # * self.mul
            if reward > self.maximum:   # record max reward
                self.maximum = reward
                print(' nHigh:{0:5.2f}'.format(self.maximum))
            # reward = np.clip( reward , -2.0, 2.0)
            # if reward > 0.1: reward = 2.0
            # elif reward < 0.00: reward = -2.0

            # if self.ok_to_render: self.ax.plot(self.bar, self.data[self.X], 'r,') 
            # if RT: self.Update_order(action)
                # wait = input("PRESS ENTER TO CONTINUE.")
                # time.sleep(3)
                # print(' nSell:{0:5.2f}'.format(reward))
                # print('sell')

        else:
            if self.already_bought:
                reward = self.data[self.X] - self.data[self.X-1]
            #     if self.ok_to_render: self.ax.plot(self.bar, self.data[self.X], 'g,') 
            # else:
            #     if self.ok_to_render: self.ax.plot(self.bar, self.data[self.X], 'm,') 
            # if RT: self.Update_order(0)

            # self.ax.plot(self.bar, -1.9, 'c,') # for debug
            # self.ax.plot(self.bar, 1.9, 'c,') # for debug
        # assert self.X < 3500

        return self._get_ob(), reward, self.done, {"bar": self.bar} # , dtype=np.float32

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
            self.ax.set_xlim(0, self.screen_width + 3)
            # self.ax.set_ylim(self.data.min(), self.data.max()) # padding done in reset
            self.ok_to_render = True
            self.viewer = rendering.SimpleImageViewer()
        img = self._get_image() # img is a numpy x,y,3 array, 210,160,3
        self.viewer.imshow(img) # display the breakout inital screen
        # wait = input("PRESS ENTER TO CONTINUE.")
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

    def Update_order(self, action): # store action in ramfs file for WL
        f = open(self.orders_file, "w") ; f.writelines(str(action)) ; f.close()
        time.sleep(0.14) # Filesystem gaurd time. allow file to close properly  
        return

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        return
