## KEY SETTINGS
#####################################

w = 4
h = 3

# Load modules
import os
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import MaxNLocator
plt.style.use('science')

os.chdir("...")

results = [
    "ihdp_1",
    "ihdp_2",
    "news_2",
    "synth_1",
    ]

for result in results:
    path = "res/res:exp_" + result + ".csv"
    data = pd.read_csv(path)

    # Rm brier score cols
    data = data[[col for col in data.columns if not col.startswith("Brier score ")]]

    # Rm mise prefix
    data.columns = [col.replace("MISE ", "") for col in data.columns]

    means = (
        data.groupby(["bias", "rm_confounding"])
        .mean()
        .round(2)
        .sort_values("rm_confounding", ascending=False)
        .sort_values("bias", ascending=True)
        .transpose()
        .sort_index()
    )

    means.drop(["seed","x_resampling"], inplace=True)

    means.columns = ["Base", "$d$ non-unif.", "$d$ conf."]

    # Transpose the DataFrame so that rows become columns and vice versa
    transposed_means = means.transpose()

    # Create a line plot
    fig = plt.figure()
    
    ax0 = fig.add_axes([0, 0, 1, 1])
    ax0.axis('off')
    
    ax1 = fig.add_axes([0.15,0.2,0.9,0.9])
    
    # Set lower lim
    #ax1.set_ylim(bottom=0)
    
    markers = ['x','x','x','x','x','x','x','o']

    for column,marker in zip(transposed_means.columns, markers):
        ax1.plot(transposed_means.index, transposed_means[column], marker=marker, label=column)

    ax1.set_ylabel('MISE', fontsize=20)
    ax1.set_xlabel('Scenario', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    fig.set_size_inches(w,h)

    plt.savefig(f"res_per_ds_{result}.pdf")

fig_leg = plt.figure(figsize=(1, 2))
ax_leg = fig_leg.add_subplot(111)

new_labels = ['Reg. tree', 'DRNet', 'GAM', 'Lin. reg.', 'MLP', 'SCIGAN', 'VCNet', 'xgboost']

# Get the handles from the previous plot
handles, _ = ax1.get_legend_handles_labels()


# Add the legend from the previous plot to the new figure
ax_leg.legend(handles, new_labels)

# Hide the axes of the new figure
ax_leg.axis('off')

fig_leg.savefig("res_per_ds_legend.pdf")
