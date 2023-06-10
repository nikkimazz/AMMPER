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


def cellPlot(data,gen,radData,ROSData,radGen,N,plots_dir):
    from matplotlib import pyplot as plt
    
    minmax = get_fixed_mins_maxs(0, N)

    # COLOR BLIND SAFE COLORS
    healthy = '#91bfdb'
    damaged = '#ffffbf'
    dead = '#fc8d59'

    n = len(data)


    # Initiating data master >>> <<<
    df_data = pd.DataFrame(data, columns=['Generation', 'x', 'y', 'z', 'Health'])


    for g in range(gen+1):
        figName = 'fig' + str(g)
        axName = 'ax' + str(g)
        #plt.style.use('seaborn-poster')
        locals()[figName] = plt.figure()
        locals()[axName] = locals()[figName].add_subplot(projection='3d')
        locals()[axName].set_xlim(minmax)
        locals()[axName].set_ylim(minmax)
        locals()[axName].set_zlim(minmax)
        locals()[axName].set_xlabel('X')
        locals()[axName].set_ylabel('Y')
        locals()[axName].set_zlabel('Z')
        plt.title('g = ' + str(g))

        #########################
        # Here we need to define our data as data_generation(secific g) ---> easier with pandas
        data = df_data.loc[df_data['Generation'] == g]
        ###############################
        # Now need to go back to numpy array without titles >>> <<<
        data1 = data.loc[data['Health'] == 1]
        data2 = data.loc[data['Health'] == 2]
        data3 = data.loc[data['Health'] == 3]

        data1 = data1.to_numpy()
        data2 = data2.to_numpy()
        data3 = data3.to_numpy()
        # print(g)
        # # axName = 'ax' + str(g)  #### <------
        # print(axName)

        locals()[axName].scatter(data1[:, 1], data1[:, 2], data1[:, 3], c=healthy, alpha=0.15, marker = 'o')
        locals()[axName].scatter(data2[:, 1], data2[:, 2], data2[:, 3], c=damaged, alpha=1, marker = '^')
        locals()[axName].scatter(data3[:, 1], data3[:, 2], data3[:, 3], c=dead, alpha=1, marker = '*')
        locals()[axName].scatter(radData[:, 0], radData[:, 1], radData[:, 2], c='#9ED9A1', alpha=1, marker = 'o')
        
    
    for g in range(gen+1):
        figName = 'fig' + str(g)
        locals()[figName].savefig(plots_dir + figName, dpi = 1200)

        # radName = 'ax' + str(radGen)
        #
    #     if type(radData) != int:
    # #     n = len(radData)
    ### radPos = [radData[i, 0], radData[i, 1], radData[i, 2]]
    ########################## >>> <<< ##################

    #    radName = 'ax' + str(radGen)
    #
    # if type(radData) != int:
    #     n = len(radData)
    #     for i in range(n):
    #         locals()[radName].scatter(radData[i,0],radData[i,1],radData[i,2],s = 10,c = 'yellow')
##### PLOT ROS, for now plot only radiation events as ROS because ROS now is difussing through medium, don't plot
# planes yet to avoid heavy computational time.


# if type(ROSData) != int:
#     n = len(ROSData)
#     for g in range(gen+1):
#         if g >= radGen:
#             ROSName = 'ax' + str(g)
#             locals()[ROSName].scatter(ROSData[:,0],ROSData[:,1],ROSData[:,2],s = 5,c = '#9ED9A1', alpha = 1)