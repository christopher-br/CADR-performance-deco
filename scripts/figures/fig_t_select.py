# LOAD MODULES

# Third party
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

####################
# Pt.2: Treatments #
####################

# Fig 3: No selection, no confounding
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Create a scatter plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

axes["A"].bar([0, 1, 2], [1/3, 1/3, 1/3], align='center', color=c_1, width=0.5)

# Set the x and y axis labels
axes["A"].set_xticks([0, 1, 2])
axes["A"].set_yticks([0, 0.5, 1])
axes["A"].set_xlabel('Treatment',fontsize=20)
axes["A"].set_ylabel('Prob.', fontsize=20)
axes["A"].set_title("Unit 1",fontsize=20)

# Show the plot
plt.savefig("t_select_00_1.pdf")

# Create a scatter plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

axes["A"].bar([0, 1, 2], [1/3, 1/3, 1/3], align='center', color=c_2, width=0.5)

# Set the x and y axis labels
axes["A"].set_xticks([0, 1, 2])
axes["A"].set_yticks([0, 0.5, 1])
axes["A"].set_xlabel('Treatment',fontsize=20)
axes["A"].set_ylabel('Prob.', fontsize=20)
axes["A"].set_title("Unit 2",fontsize=20)

# Show the plot
plt.savefig("t_select_00_2.pdf")

# Create a scatter plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

axes["A"].bar([0, 1, 2], [1/3, 1/3, 1/3], align='center', color=c_main, width=0.5)

# Set the x and y axis labels
axes["A"].set_xticks([0, 1, 2])
axes["A"].set_yticks([0, 0.5, 1])
axes["A"].set_xlabel('Treatment',fontsize=20)
axes["A"].set_ylabel('Prob.', fontsize=20)
axes["A"].set_title("Combined", fontsize=20)

# Show the plot
plt.savefig("t_select_00_com.pdf")

# Fig 4: Selection, no confounding
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Create a scatter plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

axes["A"].bar([0, 1, 2], [0.45, 0.45, 0.1], align='center', color=c_1, width=0.5)

# Set the x and y axis labels
axes["A"].set_xticks([0, 1, 2])
axes["A"].set_yticks([0, 0.5, 1])
axes["A"].set_xlabel('Treatment',fontsize=20)
axes["A"].set_ylabel('Prob.', fontsize=20)
axes["A"].set_title("Unit 1",fontsize=20)

# Show the plot
plt.savefig("t_select_10_1.pdf")

# Create a scatter plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

axes["A"].bar([0, 1, 2], [0.45, 0.45, 0.1], align='center', color=c_2, width=0.5)

# Set the x and y axis labels
axes["A"].set_xticks([0, 1, 2])
axes["A"].set_yticks([0, 0.5, 1])
axes["A"].set_xlabel('Treatment',fontsize=20)
axes["A"].set_ylabel('Prob.', fontsize=20)
axes["A"].set_title("Unit 2",fontsize=20)

# Show the plot
plt.savefig("t_select_10_2.pdf")

# Create a scatter plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

axes["A"].bar([0, 1, 2], [0.45, 0.45, 0.1], align='center', color=c_main, width=0.5)

# Set the x and y axis labels
axes["A"].set_xticks([0, 1, 2])
axes["A"].set_yticks([0, 0.5, 1])
axes["A"].set_xlabel('Treatment',fontsize=20)
axes["A"].set_ylabel('Prob.', fontsize=20)
axes["A"].set_title("Combined", fontsize=20)

# Show the plot
plt.savefig("t_select_10_com.pdf")

# Fig 5: Selection, no confounding
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

# Create a scatter plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

axes["A"].bar([0, 1, 2], [0.7, 0.2, 0.1], align='center', color=c_1, width=0.5)

# Set the x and y axis labels
axes["A"].set_xticks([0, 1, 2])
axes["A"].set_yticks([0, 0.5, 1])
axes["A"].set_xlabel('Treatment',fontsize=20)
axes["A"].set_ylabel('Prob.', fontsize=20)
axes["A"].set_title("Unit 1",fontsize=20)

# Show the plot
plt.savefig("t_select_11_1.pdf")

# Create a scatter plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

axes["A"].bar([0, 1, 2], [0.2, 0.7, 0.1], align='center', color=c_2, width=0.5)

# Set the x and y axis labels
axes["A"].set_xticks([0, 1, 2])
axes["A"].set_yticks([0, 0.5, 1])
axes["A"].set_xlabel('Treatment',fontsize=20)
axes["A"].set_ylabel('Prob.', fontsize=20)
axes["A"].set_title("Unit 2",fontsize=20)

# Show the plot
plt.savefig("t_select_11_2.pdf")

# Create a scatter plot
fig, axes = plt.subplot_mosaic("A")
fig.set_size_inches(w, h)

axes["A"].bar([0, 1, 2], [0.45, 0.45, 0.1], align='center', color=c_main, width=0.5)

# Set the x and y axis labels
axes["A"].set_xticks([0, 1, 2])
axes["A"].set_yticks([0, 0.5, 1])
axes["A"].set_xlabel('Treatment',fontsize=20)
axes["A"].set_ylabel('Prob.', fontsize=20)
axes["A"].set_title("Combined", fontsize=20)

# Show the plot
plt.savefig("t_select_11_com.pdf")