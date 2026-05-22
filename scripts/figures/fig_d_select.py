# LOAD MODULES

# Third party
from scipy.stats import beta
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

# width
w = 2
# height
h = 1

num_dots = 200
s_dot = 5
c_main = "royalblue"
c_1 = "springgreen"
c_2 = "tomato"

# Create a default random number generator
rng = default_rng()

###############
# Pt.3: Doses #
###############

# Fig 6: No selection
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Define the PDF of a uniform distribution over [0, 1]
def uniform_pdf(x):
    return np.where((x > 0) & (x < 1), 1, 0)

# Generate x values
x = np.linspace(0, 1, 1000)

# Compute the PDF
y = uniform_pdf(x)

# Create a line plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

# Create a plot
axes["A"].plot(x, y, c=c_1)

# Set the x and y axis labels
axes["A"].set_xticks([0, 0.5, 1])
axes["A"].set_yticks([0, 2, 4])
axes["A"].set_title('Unit 1', fontsize=20)
axes["A"].set_ylabel('Density', fontsize=20)
axes["A"].set_xlabel('Dose', fontsize=20)

# Save the plot
plt.savefig("d_select_00_1.pdf")

# Create a line plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

# Create a plot
axes["A"].plot(x, y, c=c_2)

# Set the x and y axis labels
axes["A"].set_xticks([0, 0.5, 1])
axes["A"].set_yticks([0, 2, 4])
axes["A"].set_title('Unit 2', fontsize=20)
axes["A"].set_ylabel('Density', fontsize=20)
axes["A"].set_xlabel('Dose', fontsize=20)

# Save the plot
plt.savefig("d_select_00_2.pdf")

# Create a line plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

# Create a plot
axes["A"].plot(x, y, c=c_main)

# Set the x and y axis labels
axes["A"].set_xticks([0, 0.5, 1])
axes["A"].set_yticks([0, 2, 4])
axes["A"].set_title('Combined', fontsize=20)
axes["A"].set_ylabel('Density', fontsize=20)
axes["A"].set_xlabel('Dose', fontsize=20)

# Save the plot
plt.savefig("d_select_00_com.pdf")

# Fig 7: Selection
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Define the PDF of a uniform distribution over [0, 1]
def beta_pdf(x):
    return 0.5 * beta.pdf(x, 2, 8) + 0.5 * beta.pdf(x, 8, 2)

# Generate x values
x = np.linspace(0, 1, 1000)

# Compute the PDF
y = beta_pdf(x)

# Create a line plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

# Create a plot
axes["A"].plot(x, y, c=c_1)

# Set the x and y axis labels
axes["A"].set_xticks([0, 0.5, 1])
axes["A"].set_yticks([0, 2, 4])
axes["A"].set_title('Unit 1', fontsize=20)
axes["A"].set_ylabel('Density', fontsize=20)
axes["A"].set_xlabel('Dose', fontsize=20)

# Save the plot
plt.savefig("d_select_10_1.pdf")

# Create a line plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

# Create a plot
axes["A"].plot(x, y, c=c_2)

# Set the x and y axis labels
axes["A"].set_xticks([0, 0.5, 1])
axes["A"].set_yticks([0, 2, 4])
axes["A"].set_title('Unit 2', fontsize=20)
axes["A"].set_ylabel('Density', fontsize=20)
axes["A"].set_xlabel('Dose', fontsize=20)

# Save the plot
plt.savefig("d_select_10_2.pdf")

# Create a line plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

# Create a plot
axes["A"].plot(x, y, c=c_main)

# Set the x and y axis labels
axes["A"].set_xticks([0, 0.5, 1])
axes["A"].set_yticks([0, 2, 4])
axes["A"].set_title('Combined', fontsize=20)
axes["A"].set_ylabel('Density', fontsize=20)
axes["A"].set_xlabel('Dose', fontsize=20)

# Save the plot
plt.savefig("d_select_10_com.pdf")

# Fig 8: Selection and confounding
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Define the PDF of a uniform distribution over [0, 1]
def beta_pdf_main(x):
    return 0.5 * beta.pdf(x, 2, 8) + 0.5 * beta.pdf(x, 8, 2)

# Generate x values
x = np.linspace(0, 1, 1000)

# Compute the PDF
y = beta.pdf(x, 2, 8)

# Create a line plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

# Create a plot
axes["A"].plot(x, y, c=c_1)

# Set the x and y axis labels
axes["A"].set_xticks([0, 0.5, 1])
axes["A"].set_yticks([0, 2, 4])
axes["A"].set_title('Unit 1', fontsize=20)
axes["A"].set_ylabel('Density', fontsize=20)
axes["A"].set_xlabel('Dose', fontsize=20)

# Save the plot
plt.savefig("d_select_11_1.pdf")

# Compute the PDF
y = beta.pdf(x, 8, 2)

# Create a line plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

# Create a plot
axes["A"].plot(x, y, c=c_2)

# Set the x and y axis labels
axes["A"].set_xticks([0, 0.5, 1])
axes["A"].set_yticks([0, 2, 4])
axes["A"].set_title('Unit 2', fontsize=20)
axes["A"].set_ylabel('Density', fontsize=20)
axes["A"].set_xlabel('Dose', fontsize=20)

# Save the plot
plt.savefig("d_select_11_2.pdf")

# Compute the PDF
y = beta_pdf_main(x)

# Create a line plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

# Create a plot
axes["A"].plot(x, y, c=c_main)

# Set the x and y axis labels
axes["A"].set_xticks([0, 0.5, 1])
axes["A"].set_yticks([0, 2, 4])
axes["A"].set_title('Combined', fontsize=20)
axes["A"].set_ylabel('Density', fontsize=20)
axes["A"].set_xlabel('Dose', fontsize=20)

# Save the plot
plt.savefig("d_select_11_com.pdf")