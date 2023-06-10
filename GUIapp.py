
import tkinter as tk

# Open window
# root = tk.Tk()
#
# # window size
# canvas = tk.Canvas(root, width = 600, height = 300)
# # canvas.grid(columnspan = 3)
#
# # Choices
# v = tk.IntVar()
#
# tk.Label(root, text ="""Choose a cell type:""", justify = tk.LEFT, padx = 20).pack()
#
# tk.Radiobutton(root, text="WildType", padx = 20, variable = v, value = 1).pack(anchor = tk.W)
#
# tk.Radiobutton(root, text="Rad51", padx = 20, variable = v, value = 2).pack(anchor = tk.W)
#
# v2 = tk.IntVar()
#
# w = tk.Scale(root, from_ = 0, to = 20, orient = 'horizontal', variable = v2).pack()
#
#
# # Access slider values of radiation
#
#
# # Close loop for window
# root.mainloop()

from tkinter import *
from tkinter import messagebox

ws = Tk()
ws.title('PythonGuides')
ws.geometry('200x200')

def viewSelected():
    choice = var.get()
    if choice == 1:
        output = "Science"

    elif choice == 2:
        output = "Commerce"

    elif choice == 3:
        output = "Arts"
    else:
        output = "Invalid selection"

    return print(str(output))


var = IntVar()
Label(ws, text ="""Choose a cell type:""", justify = tk.LEFT, padx = 20).pack()

Radiobutton(ws, text="Science", variable=var, value=1, command=viewSelected).pack()
Radiobutton(ws, text="Commerce", variable=var, value=2, command=viewSelected).pack()
Radiobutton(ws, text="Arts", variable=var, value=3, command=viewSelected).pack()

ws.mainloop()