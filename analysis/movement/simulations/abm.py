import numpy as np
from math import *
import math
import matplotlib.pyplot as plt

# number of individuals
N=100

# set to random initial conditions on (0,1)x(0,1)
xpos = np.random.uniform(0,1,N)
ypos = np.random.uniform(0,1,N)

# set to random inital headings
heading = np.random.uniform(0,2*pi,N)

randvar = 0.1

# set speed individuals move
speed = 0.01

# run for this many time steps
TIMESTEPS = 200

mvp = 10 # mean reversion parameter
mean_heading = 0;
sig = 0.3 # noise
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



# simulate individual movement
for t in range(TIMESTEPS):
    # individuals move in direction defined by heading with fixed speed
    xpos = xpos + speed*np.cos(heading)
    ypos = ypos + speed*np.sin(heading)
    
    for i in range(N):
        heading[i] = mean_heading + exp_mr*(heading[i]-mean_heading+add_noise*np.random.normal())
    #heading = heading + np.random.normal(0,randvar,N)
    
    # boundary conditions are periodic
    xpos[xpos<0]=xpos[xpos<0]+1
    xpos[xpos>1]=xpos[xpos>1]-1
    ypos[ypos<0]=ypos[ypos<0]+1
    ypos[ypos>1]=ypos[ypos>1]-1

    # plot the positions of all individuals
    plt.clf()
    plt.plot(xpos, ypos,'k.')
    plt.axes().set_aspect('equal')
    plt.draw()
    plt.pause(0.01)
