
Model Selection 
===============

Testing fit to data for various models

Non-social models
---------------

**corRandomWalk.py** No environmental or social information. Individuals assume to follow a correlated random walk with headings drawn from a wrapped cauchy distribution centred on the current heading.

**environment.py** Weighted wrapped cauchy distribution that includes previous heading and heading all other individuals took at this point computed using an averaging kernel.

Social model
------------

All social models add a social vector to the environment model. Social vector is calculated by weighting the position and heading of neighbours according to various interaction models.

**constantModel.py** Social heading is the sum of directions toward neighbours within a fixed radius and blind angle

**constantModelAlign.py** Social heading is the sum of directions toward neighbours and headings of those neighbours within a fixed radius and blind angle

**decayModel.py** Individuals as centres of attraction but social influence decays exponentially with distance

**decayModelAlign.py** Individuals as centres of attraction and alignment but social influence decays exponentially with distance

**networkModel.py** Social heading is the sum of directions toward a set number of nearest neighbours 

**networkModelAlign.py** Social heading is the weighted sum of directions toward and headings of a set number of nearest neighbours 
