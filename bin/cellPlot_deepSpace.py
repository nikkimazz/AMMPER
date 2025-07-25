# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 08:31:34 2022

@author: asingh21
"""


import pandas as pd
import numpy as np

# function to remove overlap when setting 3d axes, from stackoverflow
# https://stackoverflow.com/questions/30196503/2d-plots-are-not-sitting-flush-against-3d-axis-walls-in-python-mplot3d/41779162#41779162
def get_fixed_mins_maxs(mins, maxs):
    deltas = (maxs - mins) / 12.
    mins = mins + deltas / 4.
    maxs = maxs - deltas / 4.

    return [mins, maxs]


def cellPlot_deepSpace(data,gen,radData,ROSData,N,plots_dir):
    from matplotlib import pyplot as plt
    
    minmax = get_fixed_mins_maxs(0, N)
    
    healthy = '#91bfdb'
    damaged = "#aa0bca"
    dead = '#fc8d59'
    apoptotic = "#f2ff00"

    n = len(data)

    df_data = pd.DataFrame(data, columns=['Generation', 'x', 'y', 'z', 'Health'])

    figs = {}
    axes = {}
    
    for g in range(gen+1):
        figName = 'fig' + str(g)
        axName = 'ax' + str(g)
        figs[figName] = plt.figure()
        axes[axName] = figs[figName].add_subplot(projection='3d')
        axes[axName].set_xlim(minmax)
        axes[axName].set_ylim(minmax)
        axes[axName].set_zlim(minmax)
        axes[axName].set_xlabel('X')
        axes[axName].set_ylabel('Y')
        axes[axName].set_zlabel('Z')
        plt.title('g = '+str(g))

        #########################
        # Here we need to define our data as data_generation(secific g) ---> easier with pandas
        data = df_data.loc[df_data['Generation'] == g]
        ###############################
        # Now need to go back to numpy array without titles >>> <<<
        # data = data.to_numpy()

        # n = len(data)
        #
        # pos = data[:, 1:4]
        # col = data[:,5]
        # col2 = data[:,6]
        axName = 'ax' + str(g)  #### <------

        data1 = data.loc[data['Health'] == 1]
        data2 = data.loc[data['Health'] == 2]
        data3 = data.loc[data['Health'] == 3]
        data4 = data.loc[data['Health'] == 4]

        data1 = data1.to_numpy()
        data2 = data2.to_numpy()
        data3 = data3.to_numpy()
        data4 = data4.to_numpy()

        axName = 'ax' + str(g)  #### <------

        axes[axName].scatter(data1[:, 1], data1[:, 2], data1[:, 3], c=healthy, alpha=0.15)
        axes[axName].scatter(data2[:, 1], data2[:, 2], data2[:, 3], c=damaged, alpha=1)
        axes[axName].scatter(data3[:, 1], data3[:, 2], data3[:, 3], c=dead, alpha=1)
        axes[axName].scatter(data4[:, 1], data4[:, 2], data4[:, 3], c=apoptotic, alpha=1)




    if type(ROSData) != int:
        n = len(ROSData)
        for g in range(gen + 1):
                ROSName = 'ax' + str(g)
                axes[ROSName].scatter(ROSData[:, 0], ROSData[:, 1], ROSData[:, 2], s=5, c='#9ED9A1', alpha=1)
        
    for g in range(gen+1):
        figName = 'fig' + str(g)
        figs[figName].savefig(plots_dir + figName)




    # n = len(data)
    # for i in range(n):
    #     currG = data[i,0]
    #     pos = data[i,1:4]
    #     health = data[i,4]
    #     if health == 1:
    #         col = healthy
    #     elif health == 2:
    #         col = damaged
    #     elif health == 3:
    #         col = dead
    #     axName = 'ax' + str(currG)
    #     locals()[axName].scatter(pos[0],pos[1],pos[2],c=col)