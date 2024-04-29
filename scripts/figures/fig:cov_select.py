# LOAD MODULES

# Third party
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

# width
w = 2
# height
h = 1

####################
# Pt.1: Covariates #
####################

num_dots = 200
s_dot = 5
c_main = "royalblue"
c_1 = "springgreen"
c_2 = "tomato"

# Create a default random number generator
rng = default_rng()

# Fig 1: No covariate selection
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Generate 100 observations from a uniform distribution
x = rng.uniform(0, 1, num_dots)
y = rng.uniform(0, 1, num_dots)

# Create a scatter plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, w)
axes["A"].scatter(x, y, s=s_dot, c=c_main)

axes["A"].set_xticks([0, 1])
axes["A"].set_yticks([0, 1])

# Show the plot
plt.savefig("cov_select_0.pdf")

# Fig 2: With covariate selection
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Generate 50 observations from two different uniform distributions
x0 = rng.uniform(0.9, 1, int(0.01 * num_dots))
y0 = rng.uniform(0.0, 0.1, int(0.01 * num_dots))

x1 = rng.uniform(0, 0.7, int(0.49 * num_dots))
y1 = rng.uniform(0, 0.7, int(0.49 * num_dots))

x2 = rng.uniform(0.7, 1.0, int(0.5 * num_dots))
y2 = rng.uniform(0.3, 1., int(0.5 * num_dots))

# Combine the two sets of observations
x = np.concatenate((x0, x1, x2))
y = np.concatenate((y0, y1, y2))

# Create a scatter plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, w)
axes["A"].scatter(x, y, s=s_dot, c=c_main)

axes["A"].set_xticks([0, 1])
axes["A"].set_yticks([0, 1])

# Show the plot
plt.savefig("cov_select_1.pdf")