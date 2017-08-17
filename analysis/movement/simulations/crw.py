import numpy as np
from math import pi, sin, cos, atan2
from math import *
import math
import matplotlib.pyplot as plt
import pandas as pd

# 10 frames = 1 second
outTracks = pd.DataFrame(columns= ['frame','x','y','heading','c_id'])

trackName = 'crw/crw.csv'


    
    
# number of individuals
N=100

# set to random initial conditions on (0,1)x(0,1)
xpos = np.random.uniform(-1,1,N)
ypos = np.random.uniform(-1,1,N)

# set to random inital headings
heading = np.random.uniform(0,2*pi,N)

randvar = 0.1


# set speed individuals move
speed = 1

# run for this many time steps
TIMESTEPS = 2000

mvp = 3 # mean reversion parameter
mean_heading = 0;
sig = 1.5 # noise
dt = 1e-2 # time step
#t = np.arange(0,dt*2,dt) #             % Time vector
#x0 = 0;                 #% Set initial condition
##rng(1);                 #% Set random seed
#W = np.zeros((len(t))); #% Allocate integrated W vector
#np.random.seed(0)
#for i in range(len(t)-1):
#    W[i+1] = sqrt(exp(2*th*dt)-1)*np.random.normal()
#
#ex = np.exp(-th*t);
#x = x0*ex+mu*(1-ex)+sig*ex*W/sqrt(2*th);
#
#np.random.seed(0)

exp_mr = math.exp(-mvp*dt)
add_noise = sig*math.sqrt((math.exp(2*mvp*dt)-1)/(2*mvp))

repRad = 0.1
socWeight = 2.5
def getHeading(i):
    angle = heading[i] #xpos[i]*2*pi + ypos[i]*2*pi 
    return angle
    
    socx = 0
    socy = 0
    for j in range(N):
        if i==j:
            continue
        distij = ((xpos[i]-xpos[j])**2+ (ypos[i]-ypos[j])**2)**0.5
        if distij < repRad:
            thisAngle = math.atan2(ypos[i]-ypos[j],xpos[i]-xpos[j])
            socx = socx + cos(thisAngle)
            socy = socy + sin(thisAngle)
    
    if socx!=0 and socy!=0:
        soc_angle = atan2(socy,socx)
        angle = atan2(sin(angle) + socWeight*sin(soc_angle),cos(angle)+socWeight*cos(soc_angle))
        
        
    return math.atan2(math.sin(angle),math.cos(angle))


# simulate individual movement
for t in range(TIMESTEPS):
    print(t, heading[0])
    

    if t%500==0:    
        # set to random initial conditions on (0,1)x(0,1)
        xpos = np.random.uniform(-1,1,N)
        ypos = np.random.uniform(-1,1,N)

    
    for i in range(N):
        mean_heading = getHeading(i)
        new_angle = math.atan2(math.sin(heading[i]-mean_heading),math.cos(heading[i]-mean_heading))
        new_angle = mean_heading + exp_mr*(new_angle+add_noise*np.random.normal())
        heading[i] =math.atan2(math.sin(new_angle),math.cos(new_angle))
    #heading = heading + np.random.normal(0,randvar,N)
    
    # individuals move in direction defined by heading with fixed speed
    xpos = xpos + dt*speed*np.cos(heading)
    ypos = ypos + dt*speed*np.sin(heading)
    # boundary conditions are periodic
    # boundary conditions are periodic
#    xpos[xpos<0]=xpos[xpos<0]+2
#    xpos[xpos>2]=xpos[xpos>2]-2
#    ypos[ypos<0]=ypos[ypos<0]+2
#    ypos[ypos>2]=ypos[ypos>2]-2
    
    
    # plot the positions of all individuals
    plt.clf()
    plt.plot(xpos, ypos,'k.')
    plt.xlim([-20,20])
    plt.ylim([-20,20])
    plt.axes().set_aspect('equal')
    plt.draw()
    plt.pause(0.01)
    
    
    newcpos = pd.DataFrame(np.column_stack((np.full(N,t,dtype='int64'),xpos,ypos,heading,np.arange(0,N))), columns= ['frame','x','y','heading','c_id'])  
    outTracks = outTracks.append(newcpos,ignore_index=True )


outTracks.to_csv(trackName, index=False)    
