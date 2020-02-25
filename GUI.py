import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog

def ButtonEvent():
   messagebox.showinfo( "COMP204", "Programming Studio")
top = tk.Tk() #creates window

top.filename =filedialog.askopenfilename(initialdir = "/",title = "Select file",
                                         filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

B = tk.Button(top, text ="Click ME!", command =ButtonEvent) #  button
B.pack()
img = ImageTk.PhotoImage(Image.open(top.filename)) # image
panel = tk.Label(top, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")
top.mainloop()