import numpy as np
from math import pi
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

# simulate individual movement
for t in range(TIMESTEPS):
    # individuals move in direction defined by heading with fixed speed
    xpos = xpos + speed*np.cos(heading)
    ypos = ypos + speed*np.sin(heading)
    
    heading = heading + np.random.normal(0,randvar,N)
    
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
