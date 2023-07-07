import sys
import os
import time

for i in range(2):

    print("Sample running: " + str(i))

    # gamma rad51 basic 2.5 Gy
    os.system("python .\AMMPERBulk_GAMMAfinal.py d b a 2.5 rad51_25")

    # gamma rad51 basic 0 Gy
    os.system("python .\AMMPERBulk_GAMMAfinal.py d b a 0 rad51_0")

    # gamma rad51 basic 1 Gy
    os.system("python .\AMMPERBulk_GAMMAfinal.py d b a 30 rad51_300")


    print("WAITING 60 seconds now")
    time.sleep(60)
    print("START")
