# This Python file uses utf-8 encoding.

"""
Graphical User Interface for AMMPER v2.0

Created by Madeline Marous, in coordination with original code created by Amrita Singh and edited by Daniel Palacios.
Review README and Credits for more information.

"""
# GUI modules
import sys
from time import sleep
import subprocess

from PySide6.QtWidgets import QApplication, QWidget

from ui_form import Ui_Widget # AMMPER interface 

# AMMPER modules
import numpy as np
import random as rand
import uuid as uuid
from cellDefinition import Cell
from genTraverse_groundTesting import genTraverse_groundTesting
from genTraverse_deepSpace import genTraverse_deepSpace
from genROS import genROS
from genROSOld import genROSOld
from cellPlot import cellPlot
from cellPlot_deepSpace import cellPlot_deepSpace
from GammaRadGen import GammaRadGen
import os
import time
import pandas as pd
import time
from sklearn.model_selection import train_test_split
start_time = time.time()

# Starting boolean
display = False

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        # Confirming connection from UI layout.
        self.stackedWidget = self.ui.stackedWidget

        self.radioButton = self.ui.radioButton
        self.radioButton_2 = self.ui.radioButton_2
        self.radioButton_3 = self.ui.radioButton_3
        self.radioButton_4 = self.ui.radioButton_4
        
        self.radioButton_5 = self.ui.radioButton_5
        self.radioButton_6 = self.ui.radioButton_6

        self.radioButton_7 = self.ui.radioButton_7
        self.radioButton_8 = self.ui.radioButton_8

        self.horizontalSlider = self.ui.horizontalSlider

        self.label_7 = self.ui.label_7

        self.checkBox = self.ui.checkBox
        self.checkBox_2 = self.ui.checkBox_2

        # Connecting GUI framework to AMMPER code.
        self.ui.pushButton.clicked.connect(self.pushButton_clicked) # Launch GUI
        self.ui.pushButton_2.clicked.connect(self.pushButton_2_clicked) # Launch CLI
        self.ui.pushButton_3.clicked.connect(self.pushButton_3_clicked) # Credits
        self.ui.pushButton_4.clicked.connect(self.pushButton_4_clicked) # Run Simulation
        self.ui.pushButton_5.clicked.connect(self.pushButton_5_clicked) # Exit

        # Connecting radio buttons.
        self.radioButton.setChecked(True)
        self.radioButton.toggled.connect(self.onRadioButtonClicked)
        self.radioButton_2.toggled.connect(self.onRadioButtonClicked)
        self.radioButton_3.toggled.connect(self.onRadioButtonClicked)
        self.radioButton_4.toggled.connect(self.onRadioButtonClicked)
        
        self.radioButton_5.toggled.connect(self.onRadioButtonClicked2)
        self.radioButton_6.toggled.connect(self.onRadioButtonClicked2)

        self.radioButton_7.toggled.connect(self.onRadioButtonClicked3)
        self.radioButton_8.toggled.connect(self.onRadioButtonClicked3)

        self.checkBox.stateChanged.connect(self.fileExport)
        self.checkBox_2.stateChanged.connect(self.fileExport)

        self.horizontalSlider.valueChanged.connect(self.Slider)
        self.slider = self.horizontalSlider.value()

        # Initializing variables.

        self.radAmount = 0.0
        self.cellType = "" 
        self.radType = ""  
        self.N = 0 
        self.gen = 0
        self.ROSType = ""
        self.Gy = float(0)
        self.simDescription = ""
        self.sliderOn = True

    def pushButton_clicked(self):
        self.stackedWidget.setCurrentIndex(1)

    def pushButton_2_clicked(self):
        QApplication.exit()
        subprocess.call(["python", "AMMPERCLI.py"]) # Launch CLI

    def pushButton_3_clicked(self):
        self.stackedWidget.setCurrentIndex(4)

    def pushButton_4_clicked(self):
        if display == True:
            self.stackedWidget.setCurrentIndex(2)
            sleep(3)
            self.stackedWidget.setCurrentIndex(3)
        else: 
            self.stackedWidget.setCurrentIndex(2)
            QApplication.exit()

    def pushButton_5_clicked(self):
        QApplication.exit()

    def onRadioButtonClicked(self):
        if self.radioButton.isChecked():
            self.sliderOn = True
            self.Gy = float(self.radAmount)
            self.radType = "150 MeV Proton"
            self.gen = 15
            self.radGen = 2
            self.N = 64
            if self.Gy == 0:
                self.radData = 0
                self.ROSData = 0
        elif self.radioButton_2.isChecked():
            self.sliderOn = False
            self.radType = "GCRSim"
            self.gen = 15
            self.radGen = 2
            self.N = 64
        elif self.radioButton_3.isChecked():
            self.sliderOn = False
            self.radType = "Deep Space"
            self.gen = 15
            self.N = 300
            self.radGen = 0
            self.Gy = 0
        elif self.radioButton_4.isChecked():
            self.sliderOn = False
            self.radType = "Gamma"
            self.gen = 15
            self.radGen = 10
            self.N = 64

    def Slider(self):
        if self.sliderOn == True:
            self.slider = self.horizontalSlider.value()
            if self.slider == 1:
                self.radAmount = 0
                print('yay')
            elif self.slider == 2:
                self.radAmount = 2.5
            elif self.slider == 3:
                self.radAmount = 5
            elif self.slider == 4:
                self.radAmount = 10
            elif self.slider == 5:
                self.radAmount = 20
            elif self.slider == 6:
                self.radAmount = 30
        elif self.sliderOn == False:
            self.horizontalSlider.setEnabled(False)
        self.Gy = float(self.radAmount)

    def onRadioButtonClicked2(self):
        if self.radioButton_5.isChecked():
            self.cellType = "wt"
        elif self.radioButton_6.isChecked():
            self.cellType = "rad51"


    def onRadioButtonClicked3(self):
        if self.radioButton_7.isChecked():
            self.ROSType = "Basic ROS"
        elif self.radioButton_8.isChecked():
            self.ROSType = "Complex ROS"


    def fileExport(self):
        if self.checkBox.isChecked():
            path = self.plainTextEdit.toPlainText()
            try:
                with open(path, 'r') as file:  
                    # Download the files
                    pass
            except:
                self.label_7.setText("Invalid path.")
                self.label_7.setStyleSheet("color: red;")
        if self.checkBox_2.isChecked():
            display = True

    def simDesc(self):
        self.simDescription = "Cell Type: " + self.cellType + "\nRad Type: " + self.radType + "\nSim Dim: " + 
        str(self.N) + "microns\nNumGen: " + str(self.gen) + "ROS model: " + str(self.ROSType)

    # Amrita & Daniel's code, with Maddie's editing

    # results folder name with the time that the simulation completed
    resultsName = time.strftime('%m-%d-%y_%H-%M') + "/"
    # determine path that all results will be written to
    resultsFolder = "Results/"
    currPath = os.path.dirname("AMMPER")
    allResults_path = os.path.join(currPath,resultsFolder)
    currResult_path = os.path.join(allResults_path,resultsName)
    plots_path = os.path.join(currResult_path,"Plots/")

    # if any of the folders do not exist, create them
    if not os.path.isdir(resultsFolder):
        os.makedirs(resultsFolder)
    if not os.path.isdir(currResult_path):
        os.makedirs(currResult_path)
    if not os.path.isdir(plots_path):
        os.makedirs(plots_path)

    # write description to file
    np.savetxt(currResult_path+'simDescription.txt',[simDescription],fmt='%s')
"""
    # SIMULATION SPACE INITIALIZATION

    # cubic space (0 = no cell, 1 = healthy cell, 2 = damaged cell, 3 = dead cell)
    T = np.zeros((N,N,N),dtype= int)
    # END OF SIMULATION SPACE INITIALIZATION
    # CELL INITIALIZATION

    # first cell is at center and is healthy
    firstCellPos = [int(N/2),int(N/2),int(N/2)]

    initCellHealth = 1

    # cells = list of cells - for new cells, cells.append
    #firstCellPos = initCellPos[0,:]
    firstUUID = uuid.uuid4()
    firstCell = Cell(firstUUID,firstCellPos,initCellHealth,0,0,0,0)
    T[firstCellPos[0],firstCellPos[1],firstCellPos[2]] = firstCell.health
    cells = [firstCell]
    # data: [generation, cellPosition, cellHealth]
    data = [0,firstCell.position[0],firstCell.position[1],firstCell.position[2],firstCell.health]    


    # END OF CELL INITIALIZATION

    #placeholder initialization
    if radType == "Deep Space":
        radData = np.zeros([1,7],dtype = float)
        ROSData = np.zeros([1,7],dtype = float)

    print("Simulation beginning.")
    for g in range(1,gen+1):
        print("Generation " + str(g))
        # calculation of radiation in simulation space
        if radType == "Gamma":
            if g == radGen:
                
                dose = 1
                # radData = np.zeros([1, 6], dtype=float)
                # Dose input, radGenE stop point for gamma radiation.
                radData = GammaRadGen(dose)
                # radData = np.delete(radData, (0), axis=0)
                
                if ROSType == "Complex ROS":
                    ROSData = genROS(radData, cells)
                if ROSType == "Basic ROS":
                    ROSData = genROSOld(radData, cells)
            
            
            
        if radType == "150 MeV Proton":
            if g == radGen:
                protonEnergy = 150
                # these fluences are pre-calculated to deliver the dose to the volume of water
                if Gy != 0:
                    if Gy == 2.5:
                        trackChoice = [1]
                        energyThreshold = 0
                    elif Gy == 5:
                        trackChoice = [1,1]
                        energyThreshold = 0
                    elif Gy == 10:
                        trackChoice = [1,1,1,1]
                        energyThreshold = 0
                    elif Gy == 20:
                        trackChoice = [1,1,1,1,1,1,1,1]
                        energyThreshold = 0
                    elif Gy == 30:
                        trackChoice = [1,1,1,1,1,1,1,1,1,1,1,1]
                        energyThreshold = 0
                    
                    # placeholder initialization - will hold information on all radiation energy depositions
                    radData = np.zeros([1,6],dtype = float)
                    # ROSData = np.zeros([1,6],dtype = float)

                    for track in trackChoice:
                        trackNum = track
                        # creates a traverse for every track in trackChoice
                        radData_trans = genTraverse_groundTesting(N,protonEnergy,trackNum,energyThreshold,radType)
                        # compile all energy depositions from individual tracks together
                        radData = np.vstack([radData,radData_trans])
                    
                    #remove placeholder of 0s from the beginning of radData
                    radData = np.delete(radData,(0),axis = 0)
                                
                    # direct energy results in ROS generation - use energy depositions to calculate ROS species
                    if ROSType == "Complex ROS":
                        ROSData = genROS(radData,cells)
                    if ROSType == "Basic ROS":
                        ROSData = genROSOld(radData,cells)
                    #
                    # ROSData = np.delete(ROSData, (0), axis = 0)
                    
        elif radType == "Deep Space": 
            
            # take information from text file for #tracks of each proton energy
            deepSpaceFluenceDat = np.genfromtxt('DeepSpaceFluence0.1months_data.txt')
            # NOTE: For Deep Space sim, radiation delivery is staggered over time
            for it in range(len(deepSpaceFluenceDat)):
                # get generation at which traversal will occur
                currG = int(deepSpaceFluenceDat[it,5])
                # determine how many traversals occur at this generation
                numTrav = int(deepSpaceFluenceDat[it,4])
                if g == currG:
                    # determine what proton energy the traversal has
                    protonEnergy = deepSpaceFluenceDat[it,0]
                    # parameter that allows non-damaging energy depositions to be ignored (used to speed up simulation)
                    energyThreshold = 20
                    for track in range(numTrav):
                        # choose a random track out of the 8 available/proton energy
                        trackNum  = int(rand.uniform(0,7))
                        # generate traversal data for omnidirectional traversals
                        radData_trans = genTraverse_deepSpace(N,protonEnergy,trackNum,energyThreshold)
                        # generate ROS data from the traversal energy deposition
                        # ROSData_new = genROS(radData_trans,cells)
                        if ROSType == "Complex ROS":
                            ROSData_new = genROS(radData_trans, cells)
                        if ROSType == "Basic ROS":
                            ROSData_new = genROSOld(radData_trans, cells)
                        
                        # creates a column indicating what generation the ROS and radData occured at
                        genArr = np.ones([len(radData_trans),1],dtype=int)*g
                        # compile radData with the generation indicator
                        radData_trans = np.hstack((radData_trans,genArr))
                        # compile radData from this traversal with all radData
                        radData = np.vstack([radData,radData_trans])
                        
                        # compile ROSData with the generation indicator
                        ROSData_new = np.hstack((ROSData_new,genArr))
                        #compile ROSData with all ROSData
                        ROSData = np.vstack([ROSData,ROSData_new])
                        
        elif radType == "GCRSim":
            if g == radGen:
                # placeholder initialization
                radData = np.zeros([1,6],dtype = float)
                # take information from text file on traversals that will occur
                GCRSimFluenceDat = np.genfromtxt('GCRSimFluence_data.txt',skip_header = 1)
                for it in range(len(GCRSimFluenceDat)):
                    # for every traversal, get the proton energy of it
                    protonEnergy = int(GCRSimFluenceDat[it,0])
                    # parameter that allows non-damaging energy depositions to be ignored (used to speed up simulation)
                    energyThreshold = 20
                    # choose a random track out of the 8 available/proton energy
                    trackNum = int(rand.uniform(0,7))
                    # generate traversal data for unidirectional traversals
                    radData_trans = genTraverse_groundTesting(N,protonEnergy,trackNum,energyThreshold,radType)
                    # compile radData from this traversal with all radData
                    radData = np.vstack([radData,radData_trans])
                    
                    #remove placeholder from beginning
                    radData = np.delete(radData,(0),axis = 0)
                # generate ROS data from all traversal energy depositions
                #ROSData = genROS(radData,cells)
                if ROSType == "Complex ROS":
                    ROSData = genROS(radData, cells)
                if ROSType == "Basic ROS":
                    ROSData = genROSOld(radData, cells)

        
        # initialize list of cells that have moved
        movedCells = []
        # for every existing cell, determine whether a cell moves. If it does, write it to the list
        for c in cells:
            
            initPos = c.position
            initPos = [initPos[0],initPos[1],initPos[2]]
            movedCell = c.brownianMove(T,N,g)
            newPos = movedCell.position
            # if cell has moved
            if initPos != newPos and newPos != -1:
                # document new position in simulation space, and assign old position to empty
                T[initPos[0],initPos[1],initPos[2]] = 0
                T[newPos[0],newPos[1],newPos[2]] = c.health
                
                movedCells.append(c)

        # initialize list of new cells
        newCells = []
        # for every existing cell, determine whether a cell replicates. If it does, write the new cell to the list
        for c in cells:
            
            health = c.health
            #position = c.position
            UUID = c.UUID
            
            # cell replication
            if health == 1:
                # cellRepl returns either new cell, or same cell if saturation conditions occur
                newCell = c.cellRepl(T,N,g)
                newCellPos = newCell.position
                newCellUUID = newCell.UUID
                # if newCell the same as old cell, then saturation conditions occurred, and no replication took place
                if newCellUUID != UUID and newCellPos != -1:
                    # only document new cell if old cell replicated    
                    # if new cell is avaialble, assign position as filled                
                    T[newCellPos[0],newCellPos[1],newCellPos[2]] = 1
                    newCells.append(newCell)
        
        
        # if radiation traversal has occured
        if (radType == "150 MeV Proton" and g == radGen and Gy != 0) or (radType == "Deep Space") or (radType == "GCRSim" and g == radGen) or (radType == "Gamma" and g >= radGen):
            # initialize list of cells affected by ion/electron energy depositions
            dirRadCells = []
            for c in cells:
                health = c.health
                if cellType == "wt":
                    radCell = c.cellRad(g,radGen,radData,radType)
                elif cellType == "rad51":
                    radCell = c.cellRad_rad51(g,radGen,radData,radType)
                if type(radCell) == Cell:
                    newHealth = radCell.health
                    if health != newHealth:
                        radCellPos = radCell.position
                        T[radCellPos[0],radCellPos[1],radCellPos[2]] = newHealth
                        dirRadCells.append(radCell)
                        ######################################################################
        # if ROS generation has occured (post-radiation)
        if (radType == "150 MeV Proton" and g >= radGen and Gy != 0) or (radType == "Deep Space") or (radType == "GCRSim" and g >= radGen) or (radType == "Gamma" and g >= radGen):
            # initialize list of cells affected by ROS
            ROSCells = []
            for c in cells:
                health = c.health
                if cellType == "wt":
                    ROSCell = c.cellROS(g,radGen,ROSData)
                elif cellType == "rad51":
                    ROSCell = c.cellROS_rad51(g,radGen,ROSData)
                newHealth = ROSCell.health
                if health != newHealth:
                    ROSCellPos = ROSCell.position
                    T[ROSCellPos[0],ROSCellPos[1],ROSCellPos[2]] = newHealth
                    ROSCells.append(ROSCell)
        # if radiation has occured and cell type is NOT rad51 (cellType = wild type)
        if (radType == "150 MeV Proton" and g > radGen and Gy != 0 and cellType != "rad51") or (radType == "Deep Space" and cellType != "rad51") or (radType == "GCRSim" and g > radGen and cellType != "rad51") or (radType == "Gamma" and g >= radGen and cellType != "rad51"):
            # initialize list of cells that have undergone repair mechanisms
            repairedCells = []
            for c in cells:
                health = c.health
                if health == 2:
                    repairedCell = c.cellRepair(g)
                    newHealth = repairedCell.health
                    repairedCellPos = repairedCell.position
                    T[repairedCellPos[0],repairedCellPos[1],repairedCellPos[2]] = newHealth
                    repairedCells.append(repairedCell)
                    
        # documenting cell movement in the cells list
        for c in movedCells:
            cUUID = c.UUID
            newPos = c.position
            for c2 in cells:
                c2UUID = c2.UUID
                if cUUID == c2UUID:
                    c2.position = newPos
                
        # documenting cell replication in the cells list
        cells.extend(newCells)
        
        # documenting cell damage
        # if radiation has occurred
        if (radType == "150 MeV Proton" and g >= radGen and Gy != 0) or radType == "Deep Space" or (radType == "GCRSim" and g >= radGen) or (radType == "Gamma" and g >= radGen):
            # for every cell damaged by ion or electron energy depositions
            for c in dirRadCells:
                # get information about damaged cell
                cUUID = c.UUID
                newHealth = c.health
                newSSBs = c.numSSBs
                newDSBs = c.numDSBs
                # find cell in cell list that matches damaged cell ID
                for c2 in cells:
                    c2UUID = c2.UUID
                    if cUUID == c2UUID:
                        # adjust information about cell in cell list to reflect damage
                        c2.health = newHealth
                        c2.numSSBs = newSSBs
                        c2.numDSBs = newDSBs
            # for every cell damaged by ROS
            for c in ROSCells:
                # get information about damaged cell
                cUUID = c.UUID
                newHealth = c.health
                newSSBs = c.numSSBs
                # find cell in cell list that matches damaged cell ID
                for c2 in cells:
                    c2UUID = c2.UUID
                    if cUUID == c2UUID:
                        # adjust information about cell in cell list to reflect damage
                        c2.health = newHealth
                        c2.numSSBs = newSSBs
        
        #documenting cell repair
        # if cell can repair (not rad51), and radiation has occured
        if (radType == "150 MeV Proton" and g > radGen and Gy != 0 and cellType != "rad51") or (radType == "Deep Space" and cellType != "rad51") or (radType == "GCRSim" and g > radGen and cellType != "rad51") or (radType == "Gamma" and g >= radGen and cellType != "rad51"):
            # for every cell that has undergone repair
            for c in repairedCells:
                # get information about repaired cell
                cUUID = c.UUID
                newHealth = c.health
                newSSBs = c.numSSBs
                newDSBs = c.numDSBs
                # find cell in cell list that matches repaired cell ID
                for c2 in cells:
                    c2UUID = c2.UUID
                    if cUUID == c2UUID:
                        # adjust information about cell in cell list to reflect repair
                        c2.health = newHealth
                        c2.numSSBs = newSSBs
                        c2.numDSBs = newDSBs
            
            
                        
        # adjust data with new generational data
        # column array to denote that new data entries are at the current generation
        genArr = np.ones([len(cells),1],dtype=int)*g
        # get cell information to store in data
        # initialize cellsHealth and cellsPos with placeholders
        cellsHealth = [0]
        cellsPos = [0,0,0]
        # for each cell, get all the associated information
        for c in cells:
            currPos = [c.position]
            currHealth = [c.health]
            # record all cell positions and healths in a list, with each cell being a new row
            cellsPos = np.vstack([cellsPos,currPos])
            cellsHealth = np.vstack([cellsHealth, currHealth])
        #remove placeholder values
        cellsPos = np.delete(cellsPos,(0),axis = 0)
        cellsHealth = cellsHealth[1:]
        # compile cell information with genArr
        newData = np.hstack([genArr,cellsPos,cellsHealth])
        
        # compile new generation data with the previous data
        data = np.vstack([data,newData])
    ######################################## Random decay, lifetime ROS for complex model ################################
        if ROSType == "Complex ROS":
            if g > radGen:
                ROSDatak  , ROSData_decayed = train_test_split(ROSData, train_size = 0.5)
                # half life 1 gen = .5, half life 2 gen = .707, half life 3 gen = .7937, 20 min half life = .125
                ROSData = ROSDatak

        

    print("Calculations complete. Plotting and writing data to file.")

    # for each simulation type, write the data to a text file titled by the radType
    # for each simulation type, plot the data as 1 figure/generation
    if radType == "150 MeV Proton":
        datName = str(radAmount)+'Gy'
        dat_path = currResult_path + datName + ".txt"
        np.savetxt(dat_path,data,delimiter = ',')
        # if ROSData != 0: for 0 Gy
        cellPlot(data, gen, radData,ROSData,radGen,N,plots_path)

    elif radType == "Deep Space":
        datName = 'deepSpace'
        dat_path = currResult_path + datName + ".txt"
        np.savetxt(dat_path,data,delimiter = ',')
        cellPlot_deepSpace(data,gen,radData,ROSData,N,plots_path)
        
    elif radType == "GCRSim":
        datName = 'GCRSim'
        dat_path = currResult_path + datName + ".txt"
        np.savetxt(dat_path,data,delimiter = ",")
        cellPlot(data,gen,radData,ROSData,radGen,N,plots_path)

    elif radType == "Gamma":
        datName = 'Gamma'
        dat_path = currResult_path + datName + ".txt"
        np.savetxt(dat_path,data,delimiter = ',')
        cellPlot(data, gen, radData,ROSData,radGen,N,plots_path)



    print("Plots and data written to Results folder.") #Maddie: maybe display this in the GUI?

    print("time elapsed: {:.2f}s".format(time.time() - start_time)) """

# Widget initialization. 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
    widget.simDescription()