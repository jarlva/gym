''' https://github.com/jinfagang/rl-solution/blob/master/solve_cart_pole.py
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
RobustScaler did seem to work with 15m
Slow:RobustScaler, QuantileTransformer. not good: Normalizer works on the rows, not the columns!
https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
'''
import gym, random, shutil, os, socket, time#, datetime
from gym import spaces #, logger
from gym.utils import seeding
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler #MinMaxScaler, QuantileTransformer, ,  PowerTransformer, RobustScaler, MaxAbsScaler
# from sklearn.decomposition import PCA


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
if RT:
    import pandas as pd
    from tsfresh import extract_relevant_features #  extract_features, 
    from tsfresh.utilities.dataframe_functions import make_forecasting_frame, impute

if os.name == 'nt': 
    nt =True
    import win32file, win32con # for file attributes
    import matplotlib.pyplot as plt # https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
    # RT = True
else: nt = False


class MyrlEnv1(gym.Env): # for RL
    metadata = { 'render.modes': ['human', 'rgb_array'], 'video.frames_per_second' : 50 }
    
    def __init__(self, bars1=2):
        super(MyrlEnv1, self).__init__()
        self.bars1 = int(bars1)  #; print('bars:' + str(self.bars1))
        # self.bars2 = bars2

        sine = not True
        sine = True

        # if sine: filen = 'pure_sine_6_data_X2' # _rand
        if sine: filen = 'XBI_30_5_0_9000_X1' #'XBI_30_5_0_9000_X2' #'2_Sine_noise_large_rand_130_cxmdX1.csv'
        else:    filen = 'XBI_39.csv'
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

        file = Path('data/{}.csv'.format(filen))
        if not os.path.exists(file):
            print('Cant find ', file)
            exit()

        self.read = np.loadtxt(file, skiprows=1 , delimiter=' ', dtype=np.float64 ) # skip first 7 rows for heading and intia bad data        
        if self.read[:,0].min() > 0: self.read[:,0] -= self.read[:,0].min() - 0.001
        # if self.read[:,5].min() > 0: self.read[:,5] -= self.read[:,5].min() - 0.01
        
        # prec=(0.999) ; pca = PCA(prec)
        # a = pca.fit_transform(self.read1[:,1:])
        # self.read = np.hstack((self.read1[:,0:1], a[:,:]))

        # a = 15 ; self.read[:,1] = np.clip(self.read[:,1], -a, a) 
        # a = 4.5 # 39:5, 8.0
        # self.read[:,3] = np.clip(self.read[:,3], -3.0, 3.0) # remove spikes. Makes a huge diference
        # self.read[:,3] = self.read[:,3] / (self.read[:,3].max() - self.read[:,3].min()) * 4
        # self.read[:,4] = np.clip(self.read[:,4], -4.0, 4.0)
        # self.read[:,5] = np.clip(self.read[:,5], -0.15, 0.15)

        # temp = np.log10( self.read[:,3] ).tolist()
        # for QT: no need to factor. default 0-1 is enough
        # self.scaler = QuantileTransformer(n_quantiles=200, output_distribution='uniform', random_state=1)  #.fit_transform(self.read)  * 2 - 1
        # a = StandardScaler(with_mean=True, with_std=True).fit_transform(self.read)
        # self.read[:,3] = a[:,3]
        # fac = 1.0 ; self.scaler = MinMaxScaler(feature_range=(-fac, fac))
        self.scaler = StandardScaler(with_mean=True, with_std=True) #.fit(self.read[:,3:4])
        # self.scaler = RobustScaler()# .fit_transform(self.read)
        # self.scaler = PowerTransformer(standardize=True) #.fit_transform(self.read)
        # self.scaler = MaxAbsScaler() #.fit_transform(self.read)

        self.scaler.fit(self.read)
        rs = self.scaler.transform(self.read)
        # if RT: self.scaler = self.scaler1

        # a = 2
        # rs[:,0] = rs[:,0] / (rs[:,0].max() - rs[:,0].min()) * 2 - 1
        # rs[:,1] = rs[:,1] / (rs[:,1].max() - rs[:,1].min()) * 6
        # rs[:,2] = rs[:,2] / (rs[:,2].max() - rs[:,2].min()) * 2
        # rs[:,3] = rs[:,3] / (rs[:,3].max() - rs[:,3].min()) * 8
        # rs[:,4] = rs[:,4] / (rs[:,4].max() - rs[:,4].min()) * 7
        # rs[:,4] -= rs[:,4].min() - 0.0001 # Bad. need to keep +/-
        # if  True and nt:
        #     fig, axs = plt.subplots(2)
        #     fig.set_size_inches(10, 10)
        #     axs[0].plot( rs[:,3])
        #     # axs[1].plot(self.g1[self.X - 200 : self.X+1])  # g1
        #     axs[0].set_title('data')
        #     # axs[1].set_title('g1')
        #     # ax.set_ylim(a[:,col].min(), a[:,col].max())
        #     # rs[:,3] = np.log10( rs3] ).tolist()
        #     # axs[1].plot( temp)
        #     plt.show()
            # exit()

        if sine: # benchmark settings for sine
            self.cycle = 32 * 5 #300  #80 is fine for pure sine
            # 0:data 1:pkwd 2:pkmd 3:cxmd 4:pkm
            self.datacol = 1 ; self.data = rs[:,self.datacol]
            self.prevbars = 1 #16 best for sine large. 16 for sine large rand

            self.g1col = 2 ; self.g1 = rs[:,self.g1col:]
            self.prevbars1 = 1 #int(1.3 * eps) #10 #40 #int(1.4 * eps)

            # self.g2col = 2 ; self.g2 = self.data[:,self.g2col]
            # self.prevbars2 = 10

            # fig, axs = plt.subplots()
            # fig.set_size_inches(10, 5)
            # axs.plot( self.data[:])
            # # axs[1].plot(self.g1[self.X - 200 : self.X+1])  # g1
            # axs.set_title('data')
            # # axs[1].set_title('g1')
            # # ax.set_ylim(a[:,col].min(), a[:,col].max())
            # plt.show()
        else:
            # 0:Close 1:cxmd 2:cxd 3:pkwd 4:pkw 5:pkm 6:cx H
            # 0:Close 1:cxmd 2:cxd 3:pkwd 4:pkmd 5:pkmdir L
            self.cycle = int(32 * 5)    #128:4
            self.datacol = 1 ; self.data = self.read[:,self.datacol]
            self.prevbars = 1   # 5 best for 5, 39,pkwd. 6 for 30m pkwd
            
            self.g1col = 3 ; self.g1 = rs[:,self.g1col]
            self.prevbars1 = 1  # 5 best for 39,pkwd. 6 for 30m pkwd
            
            # fig, axs = plt.subplots()
            # fig.set_size_inches(10, 5)
            # axs.plot( self.data[:])
            # # axs[1].plot(self.g1[self.X - 200 : self.X+1])  # g1
            # axs.set_title('data')
            # # axs[1].set_title('g1')
            # # ax.set_ylim(a[:,col].min(), a[:,col].max())
            # plt.show()
            # Get last N, inclusive: a[-N:]
            # self.g2col = 5 ; self.g2 = self.read[:,self.g2col]
            # self.prevbars2 = 3

        self.maxseries = max(self.cycle, self.prevbars)  #2 * self.cycle #1 * max(self.prevbars, self.prevbars1, self.prevbars2) #, self.prevbars3)
        self.X = self.maxseries - 1  # for RT set to last one, constant. can be left always on
        self.scale = (self.data.max() - self.data.min() )

        print(file) #, 'Scale:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.scale, self.data.min(), self.data.max()) )
        # print('Read Range:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.read[ : , self.datacol].max()-self.read[ : , self.datacol].min(), self.read[ : , self.datacol].min(), self.read[ : , self.datacol].max()) )
        # print('data Range:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.scale, self.data.min(), self.data.max()) )
        # if self.prevbars1 > 0: print('  g1:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.g1.max()-self.g1.min(), self.g1.min(), self.g1.max()) )
        # if self.prevbars2 is not None: print('  g2:{0:5.2f}, Min:{1:5.2f}, Max:{2:5.2f}'.format(self.g2.max()-self.g2.min(), self.g2.min(), self.g2.max()) )
        # print('prevbars:' + str(self.prevbars) + ', datacol:' + str(self.datacol) + ', csv:' + file)

        ''' observation notes:
        very important: already_bought
        just buy price was fast, 1:14, but avg and max bars were high. Both buy price and bar were 
        slower, 1:57, than just price but avg and max bars were about `160/400 bars compared to 280/1140
        self.delta not good '''
        low = np.hstack((
            # self.data.min(),                            # Current Price
            self.g1.min(axis=0),                        # data. Gets the min of each column
            # np.full(self.prevbars, self.data.min()),     # data/price
            0,                                          # self.already_bought
            0,                                          # self.bar
            self.data.min(),                             # buy price
            ))             
        high = np.hstack((   
            # self.data.max(),                            # Current Price
            self.g1.max(axis=0),                        # data. Gets the min of each column
            # np.full(self.prevbars, self.data.max()),     # data/price
            1,                                          # self.already_bought
            self.cycle,                                 # self.bar
            self.data.max(),                             # buy price
            ))

        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.action_space = spaces.Discrete(3)  # noop/buy/sell
        self.reward_range = (-self.scale , self.scale ) 
        # self.reward_range = (-1.2  , 1.2 ) 
        self.viewer, self.state = None, None
        self.ok_to_render = False # if okrender true will be activated after X steps
        self.buybar, self.bar, self.stepcount = 0, 0, -1
        self.Xcount, self.Ocount = 0, 0
        self.realreward = list([]) ; self.buybars = list([]) ; self.barsheld = list([])
        # setup plot      
        self.screen_height=200 #200 # 100 dots per inch
        self.screen_width= (self.cycle * 2)+1
        self.seed()
        self.mark =''
        self.acc_reward = 0.0

    def _get_ob(self):
        # print(self.X)
        # Best: self.already_bought, self.delta: 2.21m
        # Adding self.bar: not good
        # removing self.already_bought and leaving self.delta: slows down a bit. 2:21m
        # self.already_bought, self.delta, self.cycle-self.bar:not good
        # if self.already_bought: 
        #     held=self.X-self.buybar
        # else: held=0.0
        
        # data = self.data[self.X - self.prevbars+1:self.X+1]
        # state = np.hstack(( data, self.already_bought, self.bar, self.price))
        # data = self.data[self.X]
        # state = np.hstack(( data, self.already_bought, self.bar, self.price))

        g1 = self.g1[self.X]
        # state = np.hstack(( data, g1, self.already_bought, self.bar, self.price))
        state = np.hstack(( g1, self.already_bought, self.bar, self.price))

        # g1 = self.g1[self.X - self.prevbars1+1:self.X+1]
        # state = np.hstack((data, g1, self.already_bought, self.bar, self.price))

        # g2 = self.g2[self.X - self.prevbars2+1:self.X+1] 
        # state = np.hstack((data, g1, g2, self.already_bought, self.bar, self.price))

        #g3 = self.g3[self.X - self.prevbars3+1:self.X+1]
        #state = np.hstack(( data, g1, g2, g3, (self.bar, self.price)))
        # if self.bar > 45: print('self.bar ' + str(self.bar))
        # assert state.size==self.observation_space.high.size, str(state.size) + " " \
        #     + str(data.size) + " " + str(g1.size) + " " + str(self.X) + " " + str(self.bar)
        return state #, dtype=np.float32)

    def reset(self):
        self.bar, self.price = 0, 0.0
        self.done, self.already_bought = False, False
        self.stepcount += 1
        # self.X = random.randint( self.maxseries - 1, self.read.__len__() - self.cycle -1) #normally enabled for training. disabled for RT

        # if not RT:
        self.X = random.randint( self.maxseries - 1, self.read.__len__() - self.cycle -1)

        # len = 200
        # self.X = random.randint( len, self.read.__len__() - self.cycle -1)
        # self.X = 1000 #; print(self.X) # for debug 0:2]) #
        # self.scaler.fit(self.read[self.X-20:self.X+self.cycle, self.datacol:self.datacol+1]) # works but unrealistic in RT
        
        # self.scaler.fit(self.read[self.X - len : self.X + 1, self.datacol:self.datacol+1])
        # self.scaler.fit(self.read[self.X - len : self.X+1, self.datacol:self.datacol+1] )   # define a tight fit, base on last N bars
        # self.scaler.fit(self.read[self.X - len : self.X+1, self.datacol:self.datacol+1] )
        # rs = self.scaler.transform(self.read[: , self.datacol:self.datacol+1] ) #* 6 #2 -1 # Fit everything based on that scale
        # rs = rs[: ,0] / (rs[self.X - len : self.X+ self.X+1, 0].max() - rs[self.X - len : self.X+ self.X+1, 0].min()) * 6
        # rs = rs / (rs[self.X - len : self.X+1].max() - rs[self.X - len : self.X+1].min()) * 6
        # rs = rs / (rs.max() - rs.min()) * 6
        # self.data = rs[:, 0]  # * 1.9 #* 0.95 # 0.7

        # print(self.data[self.X - len : self.X+ self.cycle].min(), self.data[self.X - len : self.X+ self.cycle].max())

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

        if self.ok_to_render:
            self.ax.clear()
            self.ax.set_xlim(0, self.screen_width)  # self.ax.set_ylim(self.data.min(), self.data.max())
            self.ax.axis('off')
            self.mark = ''

        return self._get_ob()
    
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
            # reward= -1 #self.data.min()
            if self.already_bought:
                Y = self.data[self.X]
                # reward = (Y - self.price) #*1.3
                pct= np.clip( ((self.read[self.X, 0] - self.read[self.buybar, 0]) / abs(self.read[self.buybar, 0]) )*100, a_min=-900, a_max=900.0)
                self.realreward.append(pct) # needed for tensorboard stats
                self.barsheld.append(self.X-self.buybar)
            else:
                pct=0.0 ; Y=0.0 ; self.buybar=self.X
                # reward= -1 #-self.scale/2 #-0.01 * self.cycle * 2  # punsih. has to be negetive
                self.barsheld.append(0)
                self.realreward.append(0) # needed for tensorboard stats
            
            if self.ok_to_render:
                self.ax.plot(self.bar, self.data[self.X], 'y,') 
                if not RT:
                    if reward < 0: 
                        self.mark = ' X'
                        self.Xcount += 1
                    elif self.read[self.X, 0] < self.read[self.buybar, 0]: 
                        self.mark = ' O'
                        self.Ocount += 1
                    avgpct=sum(self.realreward) / len(self.realreward)
                    print('End :{0:6.2f}% Bars:{1:3} Buy:{2:6.2f} Sell:{3:6.2f} dataB:{4:5.2f} dataS:{5:5.2f} Avg%:{6:5.2f}'.format(
                        pct, (self.X-self.buybar), self.read[self.buybar, 0], self.read[self.X, 0], self.price, Y, avgpct) + self.mark)
                else:
                    if self.already_bought: 
                        self.Update_order(2) # if bought, tell WL to sell(?)
                        print('End :{0:5.2f}'.format(self.read[self.X, 0]))
                    else: self.Update_order(0)

        # Buy
        elif not self.already_bought and action == 1: 
            self.already_bought = True
            self.price = self.data[self.X]
            self.buybar = self.X
            self.buybars.append(self.bar)
            
            if self.ok_to_render: 
                self.ax.plot(self.bar, self.price, 'k.')
                if RT: 
                    self.Update_order(action)
                    print('buy:{0:5.2f}'.format(self.read[self.X, 0]))
                
        # Sell
        elif self.already_bought and action == 2: # or self.data[self.X] - self.price > 0.6):
            self.done = True
            Y = self.data[self.X] 
            # reward = np.clip((Y - self.price) * 2.4 , -1.0, 1.0)
            reward = (Y - self.price)  #* 1.5 #* 1.8 #2.5
            # if reward > 12: print(Y, self.price)
            pct= np.clip( ((self.read[self.X, 0] - self.read[self.buybar, 0]) / abs(self.read[self.buybar, 0]) )*100, a_min=-900, a_max=900.0)
            self.realreward.append(pct)
            self.barsheld.append(self.X-self.buybar)

            if self.ok_to_render: 
                self.ax.plot(self.bar, self.data[self.X], 'r.')
                if not RT:
                    if reward < 0: 
                        self.mark = ' X'
                        self.Xcount += 1
                    elif self.read[self.X, 0] < self.read[self.buybar, 0]: 
                        self.mark = ' O'
                        self.Ocount += 1
                    avgpct=sum(self.realreward) / len(self.realreward)
                    print('Gain:{0:6.2f}% Bars:{1:3} Buy:{2:6.2f} Sell:{3:6.2f} dataB:{4:5.2f} dataS:{5:5.2f} Avg%:{6:5.2f}'.format( 
                        pct, (self.X-self.buybar), self.read[self.buybar, 0], self.read[self.X, 0], self.price, Y, avgpct) + self.mark)
                else:            
                    self.Update_order(action)
                    self.acc_reward +=reward
                    print('sell:{0:5.2f}'.format(self.read[self.X, 0]))

        # else:
        #     if self.already_bought:
        #         if self.ok_to_render: self.ax.plot(self.bar, self.data[self.X], 'g,')
        #     else:
        #         if self.ok_to_render: self.ax.plot(self.bar, self.data[self.X], 'm,')
        #     if RT: self.Update_order(0)

        return self._get_ob(), reward, self.done, {} #"bar": (self.X-self.buybar)} # , dtype=np.float32

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
        # read = np.loadtxt(self.data_file, skiprows=0 , delimiter=' ', dtype=np.float64) # read more than just cycle length, if needed
        read = pd.read_csv(self.data_file, sep=' ', usecols=[3], header=None) #, nrows=9900) # seems 10k and n=200 are the limits od 4GB

        n = 5
        # save_file= file + '_' + str(n) + '_' + col
        df_shift, y = make_forecasting_frame(read[3], kind="price", 
                         max_timeshift= n, rolling_direction=1)
        X2 = extract_relevant_features(df_shift, y, column_id="id",
            column_sort="time", column_value="value", show_warnings=False, n_jobs=0)
        print('X2 relevant:', X2.shape)
        X2.insert(0, 'price', y)

        self.read = read[-self.X-1: , :]   # .. but set read to just last cycle
        a = self.scaler.fit_transform(read[:,0:2]) # scale all given array..,
        # assert qt.__len__() == self.cycle, 'data.csv length must be: ' + self.cycle
        self.data = a[ : , self.datacol] # select only the last X (self.cycle in RT) as to not confuse step() -self.X-1
        # self.g1 = qt[ : , self.g1col]
        # self.g2 = qt[ : , self.g2col]
        return

    def Update_order(self, action): # store action in ramfs file for WL
        f = open(self.orders_file, "w") ; f.writelines(str(action)) ; f.close()
        time.sleep(0.10) # 0.14 Filesystem gaurd time. allow file to close properly before read by WL 
        # 0.05 failed
        return

    def render(self, mode='human'):
        # https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
        # https://matplotlib.org/2.1.1/tutorials/introductory/customizing.html#sphx-glr-tutorials-introductory-customizing-py
        # return# self.viewer.isopen
        # if self.ok_to_render: # render only after x steps
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            import matplotlib.pyplot as plt
            # time.sleep(2.3)
            self.fig, self.ax = plt.subplots(squeeze=True)#, squeeze=True
            self.fig.set_size_inches( self.screen_width/100, self.screen_height/100)  # (w, h)
            # self.fig.patch.set_facecolor('black')
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
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.PrintScore()
        return

    def PrintScore(self):
        # lastpct = 100
        # Sum = sum(self.realreward[-lastpct:])
        # Length = len(self.realreward[-lastpct:])
        # print('Avg Reward: {0:3.2f}, Sells:{1:4}, Rewards:{2:4.0f}, Total sells:{3:6.0f}'.format( 
            # Sum/Length, Length, Sum, len(self.realreward) ))
        print('Direct Losses: ' + str(round(self.Xcount / self.stepcount * 100, 2) ) + '%. Indirect: ' + str(round(self.Ocount / self.stepcount * 100, 2)) + '%')
        # print('Direct Losses: ' + str(self.Xcount) + '. Indirect: ' + str(self.stepcount) )
        return