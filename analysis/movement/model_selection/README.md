
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
