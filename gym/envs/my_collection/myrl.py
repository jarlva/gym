''' https://github.com/jinfagang/rl-solution/blob/master/solve_cart_pole.py
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
RobustScaler did seem to work with 15m
Slow:RobustScaler, QuantileTransformer. not good: Normalizer works on the rows, not the columns!
https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
supress scenetific notation: np.set_printoptions(suppress=True)
'''
import gym, random, shutil, os, socket, time#, datetime
from gym import spaces , logger
from gym.utils import seeding
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler #, MinMaxScaler #RobustScaler # #QuantileTransformer #,  # , PowerTransformer, MaxAbsScaler
gym.logger.set_level(40) # avoid line: Box bound precision lowered by casting to
# from gym.envs.classic_control import rendering
# import matplotlib.pyplot as plt
# for REST
# from flask import Flask
# from flask_restful import Api, Resource, reqparse
# app = Flask(__name__)
# api = Api(app)
# api.add_resource(User, "/")
# app.run(debug=True)

RT = not True
if os.name == 'nt': 
    nt =True
    import win32file, win32con # for file attributes
    import matplotlib.pyplot as plt # https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
    # RT = True
    os.chdir(r"C:\Users\Jake\py\files\Stable-baselines")
else: nt = False


class MyrlEnv(gym.Env): # for RL
    metadata = { 'render.modes': ['human', 'rgb_array'], 'video.frames_per_second' : 50 }
    
    def __init__(self, bars1=2):
        super(MyrlEnv, self).__init__()
        self.bars1 = int(bars1)  #; print('bars:' + str(self.bars1))
        # self.bars2 = bars2
        sine = not True
        # sine = True

        # if sine: filen = '2_Sine_noise_large.csv' # _rand
        if sine: filen = 'X1.csv' # '2_Sine_noise_large_rand.csv'
        else:    filen = 'XBI_5.csv'
        self.data_file = (r'R:\data.csv') ; self.orders_file = (r'R:\order.csv')
        pathfile = Path.cwd() ; file = str(pathfile / filen)
        if not RT: # save csv file in test dir
            a = pathfile/(r'setting.txt')
            if a.exists():
                r = open(a._str, "r")
                dirloc = Path(r.readline())
                r.close()
                shutil.copy2(str(file), str(dirloc))
                print(str(dirloc))
                os.remove(a)

        self.read = np.loadtxt(file, skiprows=1 , delimiter=' ', dtype=np.float32 ) # skip first 7 rows for heading and intia bad data        
        if self.read[:,0].min() > 0: self.read[:,0] -= self.read[:,0].min() - 0.001
        
        # remove spikes. Makes a huge diference
        if filen == 'XBI_1.csv':
            self.read[:,8] = np.clip(self.read[:,8], -2.0, 2.0)
        elif filen == 'XBI_3.csv':
            # self.read[:,3] = np.clip(self.read[:,3], -7.0, 7.0)
            pass
        elif filen == 'XBI_5.csv':
            a= 19 ; self.read[:,a] = np.clip(self.read[:,a], -4.8 , 4.8 ) # pkwd
            a= 20 ; self.read[:,a] = np.clip(self.read[:,a], -8.5 , 8.5 )
            pass
        elif filen == 'XBI_10.csv':
            a= 19 ; self.read[:,a] = np.clip(self.read[:,a], -4.8 , 4.8 ) # pkwd
            a= 20 ; self.read[:,a] = np.clip(self.read[:,a], -8.5 , 8.5 )
            pass
        elif filen == 'XBI_15.csv':
            a= 19 ; self.read[:,a] = np.clip(self.read[:,a], -5.5 , 5.5 )
            a= 20 ; self.read[:,a] = np.clip(self.read[:,a], -8.5 , 8.5 )
            pass
        elif filen == 'XBI_30.csv':
            a= 3 ; self.read[:,a] = np.clip(self.read[:,a], -8.0 , 8.0 )  # cxd
            a= 19 ; self.read[:,a] = np.clip(self.read[:,a], -4.8 , 4.8 ) # pkwd

        # temp = np.log10( self.read[:,3] ).tolist()
        # for QT: no need to factor. default 0-1 is enough
        # qt = QuantileTransformer(n_quantiles=200, output_distribution='uniform', random_state=1).fit_transform(self.read) * 2 - 1
        # qt = self.scaler.transform(self.read)
        # a = StandardScaler(with_mean=True, with_std=True).fit_transform(self.read)
        # fac = 1.0 ; self.scaler = MinMaxScaler(feature_range=(-fac, fac))
        rs = StandardScaler(with_mean=True, with_std=True).fit_transform(self.read)
        # rs = RobustScaler().fit_transform(self.read)
        # self.scaler = PowerTransformer(standardize=True) #.fit_transform(self.read)
        # self.scaler = MaxAbsScaler() #.fit_transform(self.read)
        # minmax= MinMaxScaler(feature_range=(0, 6)).fit_transform(self.read) - 3.0
        # rs = self.scaler.transform(self.read)
        # self.scaler = StandardScaler(with_mean=True, with_std=True)

        # if RT: self.scaler = self.scaler1

        # g1_low = 1 ; g1_high = 1 ; # self.g1 = self.read[:, g1_low:g1_high+1] ; self.prevbars1 = 1
        # minmax = MinMaxScaler(feature_range=(0.0, 45.0), copy=True ).fit_transform(rs)
        # self.read[:,g1_low:g1_high+1] = np.clip(self.read[:,g1_low:g1_high+1], -cap, cap)
        if sine: # benchmark settings for sine
            self.cycle = 32 * 4 #300  #80 is fine for pure sine
            self.datacol = 0 ; self.data = self.read[:,self.datacol]
            self.prevbars = 5 #16 best for sine large. 16 for sine large rand
            self.g1col = 1 ; self.g1 = rs[:,self.g1col] ; self.prevbars1 = 16 #int(1.3 * eps) #10 #40 #int(1.4 * eps)
        else:
            # 3 cxd, 16 rsi, 19 pkwd, 22 ADb, 23 rsihigh, 24 CloseB
            self.cycle = 90 #5:280, 10m:70, 15m:60, 3m:150, 5:120m
            self.datacol = 16 ; self.data = rs[:,self.datacol] * 1 ; self.prevbars = 1 # - rs[:,self.datacol].min()
            self.g1col = 19 ; self.g1 = rs[:,self.g1col] ; self.prevbars1 = 1
            self.g2col = 23 ; self.g2 = rs[:,self.g2col] ; self.prevbars2 = 1
            self.g3col = 22 ; self.g3 = self.read[:,self.g3col] ; self.prevbars3 = 1

        self.scale = (self.data.max() - self.data.min() ) * .8
        self.maxseries = max(self.cycle, self.prevbars)
        self.X = self.maxseries - 1  # for RT set to last one, constant. can be left always on
        # print(file) #, 'Scale:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.scale, self.data.min(), self.data.max()) )
        print('prevbars:' + str(self.prevbars) + ', datacol:' + str(self.datacol) + ', csv:' + file)
        print('Read Range:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.read[ : , self.datacol].max()-self.read[ : , self.datacol].min(), self.read[ : , self.datacol].min(), self.read[ : , self.datacol].max()) )
        print('data Range:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.scale, self.data.min(), self.data.max()) )
        if self.prevbars1 > 0: print('        g1:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.g1.max()-self.g1.min(), self.g1.min(), self.g1.max()) )
        if self.prevbars2 > 0: print('        g2:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.g2.max()-self.g2.min(), self.g2.min(), self.g2.max()) )
        self.viewer, self.state = None, None ; self.action_space = spaces.Discrete(3)  # noop/buy/sell
        self.action_space.seed(5) ; self.seed(seed=5) ; self.ok_to_render = False
        self.buybar, self.bar, self.stepcount = 0, 0, -1 ; self.newmax = 0.5
        self.Xcount, self.Ocount  = 0, 0 ; self.ends, self.didntbuy = 0, 0 ; self.acc_reward = 0.0
        self.realreward = list([]) ; self.buybars = list([]) ; self.barsheld = list([]) ; self.mark ='' 
        self.screen_height=200 ; self.screen_width= (self.cycle)+1 #200 # 100 dots per inch
        
        # self.rsilow = -1.5 ; self.rsihigh = 3.8
        low = np.hstack((0., 0.,                       # self.already_bought, self.bar
            # -self.scale,                                # self.data[self.X] - self.price
            np.full(self.prevbars, self.data.min()),   # data/price
            np.full(self.prevbars1, self.g1.min()),   # g1
            np.full(self.prevbars2, self.g2.min()),     # g2
            # np.full(self.prevbars3, self.g3.min()),     # g3
            ))             
        high = np.hstack((1., self.cycle,             # self.already_bought, self.bar
            # self.scale,                         # self.data[self.X] - self.price
            np.full(self.prevbars, self.data.max()),   # data/price
            np.full(self.prevbars1, self.g1.max()),    # g1
            np.full(self.prevbars2, self.g2.max()),    # g2
            # np.full(self.prevbars3, self.g3.max()),    # g3
            )) ; self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def _get_ob(self):
        '''Best: self.already_bought, self.delta
        bar+1 helps reduce cycle=bar condition
        RSI: needs just 1 rs val bar. 5 bars slows a bit. delta, scale not needed. Per step reward faster
        price: needs the delta between X and buybar. reward is per step
        '''
        # tmp = self.data[self.X] - self.data[self.X-1]
        # tmp = self.data[self.X] - self.price
        # tmp = np.clip(self.data[self.X] - self.price, -self.scale, self.scale)
        # if self.already_bought and abs(tmp) > self.newmax:
        #     self.newmax = abs(tmp)
        #     print(tmp)
        # if self.already_bought:
        #     tmp = (self.data[self.X] - self.price) / self.price
        # else: tmp = 0.
        return np.hstack((self.already_bought, self.bar+1, 
            # self.data[self.X] - self.price,
            # self.data[self.X] - self.data[self.X-1],
            self.data[self.X],
            # self.data[self.X - self.prevbars+1:self.X+1],
            # self.g1[self.X, :],
            self.g1[self.X],
            self.g2[self.X],
            # self.g3[self.X],
            # self.g1[self.X] - self.g1[self.buybar],
            # self.g1[self.X] - self.g1[self.X-1]
            # self.data[self.X] - self.data[self.X-5],
            # self.data[self.X-1] - self.data[self.X-2],
            # self.data[self.X-2] - self.data[self.X-3],
            # self.data[self.X-3] - self.data[self.X-4],
            # self.data[self.X-4] - self.data[self.X-5]
            ))

    def step(self, action):     # **********************************************
        reward = 0.0 ; self.bar += 1
        
        self.X += 1 # active only in training for speedup
        # TS remarked for training speed
        # if not RT: self.X += 1 # remark for training speed
        # else: self.read_data() # wait for data file to update
            # if self.ok_to_render: self.ax.plot(self.bar, 0, 'c,') # for debug

        # self.scaler.fit(self.read[self.X, self.datacol:self.datacol+1])   # define a tight fit, base on last N bars
        # rs = self.scaler.transform(self.read[self.X:self.X+1, self.datacol:self.datacol+1] ) #* 5.4 # Fit everything based on that scale
        # rs = self.scaler.transform(self.read[self.X, self.datacol:self.datacol+1] )
        # rs[:,0] = rs[:,0] / (rs[:,0].max() - rs[:,0].min()) * 6
        # self.data[self.X] = rs[0] #* 6 #2 -1
        # if rs[:,0].max() - rs[:,0].min() > 6: print(rs[0])

        # # reached end
        if self.bar == self.cycle:
            self.done = True # end of episode
            self.ends += 1
            reward = self.data[self.X] - self.data[self.X-1]
            if self.already_bought:
                
                # reward = -5.0
                pct= np.clip( ((self.read[self.X, 0] - self.read[self.buybar, 0]) / abs(self.read[self.buybar, 0]) )*100, a_min=-20.0, a_max=20.0)
            else:
                pct=0.0 ; self.buybar=self.X ; self.buybars.append(0)
                self.didntbuy += 1 #; print('didnt buy')

            self.realreward.append(pct) # needed for tensorboard stats
            self.barsheld.append(self.X-self.buybar)
            
            if self.ok_to_render:
                self.ax.plot(self.bar, self.data[self.X], 'y,') 
                if not RT:
                    if self.data[self.X] <= self.price: # and self.read[self.X, 0] < self.read[self.buybar, 0]: #self.data[self.buybar]: 
                        self.mark = ' X'
                        self.Xcount += 1
                    elif pct <= 0: #elif self.read[self.X, 0] < self.read[self.buybar, 0]: 
                        self.mark = ' O'
                        self.Ocount += 1
                    avgpct=sum(self.realreward) / len(self.realreward)

                    print('End :{0:6.2f}% BBar:{1:3} Buy:{2:6.2f} Sell:{3:6.2f} dataB:{4:5.2f} dataS:{5:5.2f} Avg%:{6:3.1f}'.format(
                        pct, self.buybars[-1], self.read[self.buybar, 0], 
                        self.read[self.X, 0], self.price, self.data[self.X], avgpct ) + self.mark)
                else:
                    if self.already_bought: 
                        self.Update_order(2) # if bought, tell WL to sell(?)
                        print('End :{0:5.2f}'.format(self.read[self.X, 0]))
                    else: self.Update_order(0)
                
                self.mark = ''
        # Buy
        elif not self.already_bought and action == 1:
            self.price = self.data[self.X]
            self.already_bought = True ; self.buybar = self.X ; self.buybars.append(self.bar)
            
            if self.ok_to_render: 
                self.ax.plot(self.bar, self.price, 'w,')
                if RT: 
                    self.Update_order(action)
                    print('buy:{0:5.2f}'.format(self.read[self.X, 0]))
        # Sell
        elif self.already_bought and (action == 2): # or self.data[self.X] - self.price > 1.) : #(self.data[self.X] - self.price) / self.price > .01 ): #0.02 of 6 = 0.0033% of 6 
            self.done = True
            # reward = self.data[self.X] - self.price
            # reward = self.price, self.data[self.X]
            pct = np.clip( ((self.read[self.X, 0] - self.read[self.buybar, 0]) / abs(self.read[self.buybar, 0]) )*100, a_min=-20.0, a_max=20.0) # self.pctdiff(self.read[self.buybar, 0], self.read[self.X, 0]) #
            self.realreward.append(pct)
            self.barsheld.append(self.X-self.buybar)

            if self.ok_to_render: 
                self.ax.plot(self.bar, self.data[self.X], 'r,')
                if not RT:
                    if self.data[self.X] <= self.price: # and self.read[self.X, 0] < self.read[self.buybar, 0]:
                        self.mark = ' X'
                        self.Xcount += 1
                    elif pct <= 0: #self.read[self.X, 0] < self.read[self.buybar, 0]: 
                        self.mark = ' O'
                        self.Ocount += 1
                    avgpct=sum(self.realreward) / len(self.realreward)
                    # Y = self.data[self.X]
                    print('Gain:{0:6.2f}% BBar:{1:3} Buy:{2:6.2f} Sell:{3:6.2f} dataB:{4:5.2f} dataS:{5:5.2f} Avg%:{6:3.1f}'.format( 
                        pct, self.buybars[-1], self.read[self.buybar, 0],
                        self.read[self.X, 0], self.price, self.data[self.X], avgpct ) + self.mark)
                else:
                    self.Update_order(action)
                    self.acc_reward +=reward
                    print('sell:{0:5.2f}'.format(self.read[self.X, 0]))
                
                self.mark = ''
        else:
            if self.already_bought:
                reward = self.data[self.X] - self.data[self.X-1]
        #         if self.ok_to_render: self.ax.plot(self.bar, self.data[self.X], 'y,')
        #     else:
        #         if self.ok_to_render: self.ax.plot(self.bar, self.data[self.X], 'm,')
            # if RT: self.Update_order(0)

        return self._get_ob(), reward, self.done, {} #"bar": (self.X-self.buybar)} # , dtype=np.float32

    def reset(self):
        self.bar, self.buybar, self.price = 0, 0, 0.0
        self.done, self.already_bought = False, False
        self.stepcount += 1

        # self.X = random.randint( self.maxseries - 1, self.read.__len__() - self.cycle -1) #normally enabled for training. disabled for RT

        # if not RT:
        self.X = random.randint( self.maxseries - 1, self.read.__len__() - self.cycle -1) #; print('random reset:' + str(self.X)) #; exit()

        # len = 200
        # self.X = random.randint( len, self.read.__len__() - self.cycle -1)
        # self.X = 1000 #; print(self.X) # for debug 0:2]) #
        # self.scaler.fit(self.read[self.X-20:self.X+self.cycle, self.datacol:self.datacol+1]) # works but unrealistic in RT

        # self.scaler.fit(self.read[self.X - self.cycle : self.X + 1, self.datacol:self.datacol+1])
        # self.scaler.fit(self.read[self.X - len : self.X + 1, self.datacol:self.datacol+1])
        # self.scaler.fit(self.read[self.X - len : self.X+1, self.datacol:self.datacol+1] )   # define a tight fit, base on last N bars
        # self.scaler.fit(self.read[self.X - len : self.X+1, self.datacol:self.datacol+1] )
        # rs = self.scaler.transform(self.read[: , self.datacol:self.datacol+1] ) #* 6 #2 -1 # Fit everything based on that scale
        # rs = rs[: ,0] / (rs[self.X - len : self.X+ self.X+1, 0].max() - rs[self.X - len : self.X+ self.X+1, 0].min()) * 6
        # rs = rs / (rs[self.X - len : self.X+1].max() - rs[self.X - len : self.X+1].min()) * 6
        # rs = rs / (rs.max() - rs.min()) * 6
        # self.data = rs[:, 0]  # * 1.9 #* 0.95 # 0.7

        # print(self.data[self.X - len : self.X+ self.cycle].min(), self.data[self.X - len : self.X+ self.cycle].max())
        
        # minmax= MinMaxScaler(feature_range=(0, 6)).fit_transform(self.read) - 3.0

        # len = 64
        # self.scaler.fit(self.read[self.X - len : self.X+1, self.g1col:self.g1col+1])   # define a tight fit, base on last N bars
        # a = self.scaler.transform(self.read[:, self.g1col:self.g1col+1])    # Fit everything based on that scale
        # self.g1 = a[:, 0]  * 0.9
        # print(self.g1[self.X - len : self.X+1].min(), self.g1[self.X - len : self.X+1].max())

        # fig, axs = plt.subplots(2)
        # fig.set_size_inches(10, 10)
        # axs[0].plot( self.data[self.X - len : self.X+self.cycle])
        # # axs[0].set_ylim(self.data[self.X - len : self.X+ self.cycle].min(), self.data[self.X - len : self.X+ self.cycle].max())
        # axs[0].set_title(str(self.datacol))
        # # axs[1].plot(self.g1[self.X - len : self.X+1])
        # # axs[1].set_ylim(self.g1[self.X - len : self.X+1].min(), self.g1[self.X - len : self.X+1].max())
        # # axs[1].set_title(str(self.g1col))
        # plt.show()

        # if self.ok_to_render:
        #     self.ax.clear()
        #     self.ax.set_xlim(0, self.screen_width)  # self.ax.set_ylim(self.data.min(), self.data.max())
        #     self.ax.axis('off')

        return self._get_ob()

    def read_data(self):
        # time.sleep(0.15)
        while True: # Wait for new data bar from WL
            # time.sleep(0.01)
            fattrs = win32file.GetFileAttributes(self.data_file)
            if fattrs & win32con.FILE_ATTRIBUTE_ARCHIVE:
                win32file.SetFileAttributes(self.data_file, 128) # reset archive bit
                time.sleep(0.05)    # 0.05
                break
            time.sleep(0.03)  # 0.03
        read = np.loadtxt(self.data_file, skiprows=0 , delimiter=' ', dtype=np.float64) # read more than just cycle length, if needed
        self.read = read[-self.X-1: , :]   # .. but set read to just last cycle
        a = self.scaler.fit_transform(read[:,0:2]) # scale all given array..,
        # assert qt.__len__() == self.cycle, 'data.csv length must be: ' + self.cycle
        self.data = a[ : , self.datacol] # select only the last X (self.cycle in RT) as to not confuse step() -self.X-1
        # self.g1 = qt[ : , self.g1col]
        # self.g2 = qt[ : , self.g2col]

    def Update_order(self, action): # store action in ramfs file for WL
        f = open(self.orders_file, "w") ; f.writelines(str(action)) ; f.close()
        time.sleep(0.10) # 0.14 Filesystem gaurd time. allow file to close properly before read by WL 
        # 0.05 failed

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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)  # https://pynative.com/python-random-seed/
        np.random.seed(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            self.PrintScore()

    def PrintScore(self):
        # lastpct = 100
        # Sum = sum(self.realreward[-lastpct:])
        # Length = len(self.realreward[-lastpct:])
        # print('Avg Reward: {0:3.2f}, Sells:{1:4}, Rewards:{2:4.0f}, Total sells:{3:6.0f}'.format( 
            # Sum/Length, Length, Sum, len(self.realreward) ))
        print('Direct Losses: ' + str(int(self.Xcount / self.stepcount * 100) ) + '%. Indirect: ' 
        + str(int(self.Ocount / self.stepcount * 100)) + '%'
        + '  Ends: ' + str(int(self.ends / self.stepcount * 100)) + '%'
        )
        # print('Direct Losses: ' + str(self.Xcount) + '. Indirect: ' + str(self.stepcount) )