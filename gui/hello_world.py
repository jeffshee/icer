# Python tkinter hello world program

from tkinter import *
from tkinter import font

root = Tk()
root.tk.call('tk', 'scaling', 4.0)
default_font = font.nametofont("TkDefaultFont")
default_font.configure(size=9)
a = Label(root, text="Hello World")
a.pack()
root.mainloop()
