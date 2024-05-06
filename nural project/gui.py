import tkinter as tk
from tkinter import PhotoImage


root = tk.Tk()
root.title("background image")
root.geometry('500x500')


image = PhotoImage(file="E:/nural project/background.jpg")

bg_label = tk.Label(root)
bg_label.place(relwidth=1, relheight=1)

# bg=PhotoImage(file="E:/nural project/4.png")
# ll= tk.Label(image=bg,width=2000,height=2000).pack()



root.mainloop()



