
import numpy as np
from math import pi, sin, cos, atan2
from math import *
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d



outTracks = pd.DataFrame(columns= ['frame','x','y','heading','c_id','clip'])

trackName = 'environment/data_sim.csv'


clip = 0

for clip in range(10):    
    print(clip)
    # number of individuals
    N=10
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # first we create a stochastic environment
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mvp = 1 # mean reversion parameter
    sig = 0.5 # noise
    dt = 1e-3 # time step
    exp_mr = math.exp(-mvp*dt)
    add_noise = sig*math.sqrt((math.exp(2*mvp*dt)-1)/(2*mvp))
    
    dx=0.1
    xspace = 100
    
    xn = int(xspace/dx)
    
    randField = np.zeros(xn)
    xvals = np.arange(0,xspace,dx)
    
    for x in range(1,xn):
        new_rand = randField[x-1]-0
        randField[x] = 0 + exp_mr*(new_rand+add_noise*np.random.normal())        
    
    sm = 25
    ma = np.ones(sm)/sm
    randField = randField*5/np.max(randField)
    randField = np.convolve(randField, ma, mode='same')
        
    
    
    plt.figure()
    plt.plot(xvals,randField)
    
    # this creates the interpolation function
    f_interp = interp1d(xvals, randField, kind='cubic')
    

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # put all individuals on the trail
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    xpos = np.random.uniform(0,10,N)
    ypos = np.zeros_like(xpos)
    heading = np.zeros_like(xpos)
    for i in range(N):
        ypos[i] = f_interp(xpos[i])
        
    plt.plot(xpos,ypos,'.')
    
    # run for this many time steps
    TIMESTEPS = 5000
    
    def getHeading(i):
        # function to calculate the landscape heading
        dy = -ypos[i] + f_interp(xpos[i])
        dx = 0.05
        angle = atan2(dy,dx)
        return math.atan2(math.sin(angle),math.cos(angle))
     
    
    speed=1
    
    
    # simulate individual movement
    for t in range(TIMESTEPS):
    
        for i in range(N):
            new_angle = getHeading(i)
            heading[i] =math.atan2(math.sin(new_angle),math.cos(new_angle))

        xpos = xpos + dt*speed*np.cos(heading)
        ypos = ypos + dt*speed*np.sin(heading)
        
    #=====plot if necessary ========================================================
#        if (t% 5)==0:
#            if np.min(xpos)<20:
#                  ##plot the positions of all individuals
#                plt.clf()
#                plt.plot(xpos, ypos,'k.')
#                 
#                plt.xlim([0,20])
#                plt.ylim([-6,6])
#                plt.axes().set_aspect('equal')
#                plt.draw()
#                plt.pause(0.01)
#    #==============================================================================
        
        outTime = np.full(N,t,dtype='int64')
        clipper = np.full(N,clip,dtype='int64')
        outID = np.arange(0,N)
        outputInd = (xpos>10) & (xpos<100)
        newcpos = pd.DataFrame(np.column_stack((outTime[outputInd],xpos[outputInd],ypos[outputInd],heading[outputInd],outID[outputInd],clipper[outputInd])), columns= ['frame','x','y','heading','c_id','clip'])  
        outTracks = outTracks.append(newcpos,ignore_index=True )
    

outTracks.to_csv(trackName, index=False)    
