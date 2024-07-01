import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import maxwell, skewnorm
from scipy.integrate import odeint
from statsmodels.stats.anova import AnovaRM
import pingouin as pg

def Growthcurves(results, title):

    Growth_curve = []
    Generations = []

    for i in results["Generation"].unique():

        subdf = results.loc[results["Generation"] == i]

        healthy = subdf.loc[subdf["Health"] == 1]
        unhealthy = subdf.loc[subdf["Health"] != 1]

        Healthy_counts = healthy.shape[0]
        Unhealthy_counts = unhealthy.shape[0]

        if healthy.empty:
            Healthy_counts = 0

        if unhealthy.empty:
            Unhealthy_counts = 0

        try:
            Gi = Unhealthy_counts/Healthy_counts
        except:
            # there are 0 healthy cells
            Gi = 0

        # Growth_curve ratio
        Growth_curve = np.append(Growth_curve, Gi)

        Generations = np.append(Generations, i)

    return Growth_curve, Generations

def Means_and_variances(name):

    name = name

    Growth_m = []

    for root, dirs, files in os.walk(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\Results_Bulk" + '\\' + str(name)):
         #for file in files:
             if len(files) > 1:

                 namess = ["Generation", "x", "y", "z", "Health"]

                 resultsi = pd.read_csv(os.path.join(root, files[0]), names=namess)

                 Growthi, Geni = Growthcurves(resultsi, 'Basic rad51 2.5 Gy')

                 Growth_m.append(Growthi)

    Growth_m = np.stack(Growth_m, axis = 0)
    mean =  np.mean(Growth_m, axis=0)
    std = np.std(Growth_m, axis=0)
    n = np.shape(Growth_m)[0]

    std = std / np.sqrt(n)
    Growth_m = Growth_m[:,-4:]

    gf_m = pd.DataFrame(Growth_m, columns = ['t12', 't13', 't14','t15'])
    gf_m = pd.melt(gf_m, value_vars = ['t12', 't13', 't14','t15'])
    gf_m.to_csv(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\StatisticsBulk" + "\\" + str(name) + "pivoted", sep = '\t')
    np.savetxt(r"C:\Users\danie\Desktop\Daniel_Master_Directory\AMMPER_Out\StatisticsBulk" + "\\" + str(name), Growth_m, delimiter = ',')

    return mean, std, n, Growth_m


mean, std, n, Growth_m = Means_and_variances("WT_Basic_1")

mean2, std2, n2, Growth_m2  = Means_and_variances("WT_Basic_2")

mean3, std3, n3, Growth_m3  = Means_and_variances("WT_Complex_1")

mean4, std4, n4, Growth_m4 = Means_and_variances("WT_Complex_2")

# Trim arrays to have the same shape (24, 4)
Growth_m = Growth_m[:24, :]
Growth_m2 = Growth_m2[:24, :]
Growth_m3 = Growth_m3[:24, :]
Growth_m4 = Growth_m4[:24, :]

# Now all arrays should have shape (24, 4)
column1_Growth_m = Growth_m[:, 0]
column1_Growth_m2 = Growth_m2[:, 0]
column1_Growth_m3 = Growth_m3[:, 0]
column1_Growth_m4 = Growth_m4[:, 0]
color1 = "#d31e25"
color2 = "#d7a32e"
color3 = "#369e4b"
color4 = "#5db5b7"
color5 = "#31407b"
color6 = "#d1c02b"
color7 = "#8a3f64"
color8 = "#4f2e39"

# Plotting histograms with different colors
plt.hist(column1_Growth_m, bins=20, alpha=0.5, label='Growth_m', color=color1)
plt.hist(column1_Growth_m2, bins=20, alpha=0.5, label='Growth_m2', color=color2)
plt.hist(column1_Growth_m3, bins=20, alpha=0.5, label='Growth_m3', color=color3)
plt.hist(column1_Growth_m4, bins=20, alpha=0.5, label='Growth_m4', color=color4)

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of 4 Distributions')
plt.legend()
plt.savefig('paperfigures2024/nonparametric_distributions.svg', format='svg')

