from tkinter import *
from tkinter import ttk
from tkinter.ttk import Combobox
#from tkinter import PhotoImage

from PIL import Image, ImageTk

pro = Tk()
pro.geometry('930x790+300+1')
pro.resizable(False, False)
pro.title('Service cancellation predictor')
pro.config(background='white')
pro.minsize(500, 500)

root = Tk()

canv = Canvas(root, width=80, height=80, bg='white')
canv.grid(row=2, column=3)

# PIL solution

img=ImageTk.PhotoImage (Image.open("2.jpg"))
canv.create_image (20, 20, anchor=NW, image=img)

mainloop()
bg=PhotoImage(file="F:/pythonProject9/pngtree-2018-white-8.png")
ll= Label(image=bg,width=2000,height=2000).pack()



fr = Frame(width='1250', height='100', bg='Teal')#grey  Teal
fr.pack()
# frr = Frame(width='1250', height='100', bg='Teal')
# frr.place(x=0, y=650)
v = IntVar()
g = IntVar()
com1=Combobox(pro,values=['Are', 'Perimeter','MajorAxisLength', 'MinorAxisLength', 'roundness'])
com1.place(x=260, y=150)
lbl1 = Label(pro, text='First Feature', fg='black', bg='#c3ecf3', font=(NONE, 13, 'bold'))
lbl1.place(x=120, y=150)

com2=Combobox(pro,values=['Are', 'Perimeter','MajorAxisLength', 'MinorAxisLength', 'roundness'])
#com.pack()
com2.place(x=660, y=150)
lbl2 = Label(pro, text='Second Feature', fg='black', bg='#c3ecf3', font=(NONE, 13, 'bold'))
lbl2.place(x=500, y=150)

com3=Combobox(pro,values=['Bomay','Cali','Sira'])
com3.place(x=260, y=260)
lbl3 = Label(pro, text='First Class', fg='black', bg='#c3ecf3', font=(NONE, 13, 'bold'))
lbl3.place(x=120, y=260)

com4=Combobox(pro,values=['Bomay','Cali','Sira'])
com4.place(x=660, y=260)
lbl4 = Label(pro, text='Second Class', fg='black', bg='#c3ecf3', font=(NONE, 13, 'bold'))
lbl4.place(x=500, y=260)

en5=Entry(pro,relief="flat",highlightthickness=1,highlightbackground="gray",highlightcolor="cyan",bg='white')
en5.place(x=260, y=370)
lbl5 = Label(pro, text='Learning Rate', fg='black', bg='#c3ecf3', font=(NONE, 13, 'bold'))
lbl5.place(x=120, y=370)

en6=Entry(pro,relief="flat",highlightthickness=1,highlightbackground="gray",highlightcolor="cyan",bg='white')
en6.place(x=660, y=370)
lbl6 = Label(pro, text='Number of epochs', fg='black', bg='#c3ecf3', font=(NONE, 13, 'bold'))
lbl6.place(x=500, y=370)

en7=Entry(pro,relief="flat",highlightthickness=1,highlightbackground="gray",highlightcolor="cyan",bg='white')
en7.place(x=260, y=480)
lbl7 = Label(pro, text='MSE threshold', fg='black', bg='#c3ecf3', font=(NONE, 13, 'bold'))
lbl7.place(x=120, y=480)

en8=Entry(pro,relief="flat",highlightthickness=1,highlightbackground="gray",highlightcolor="cyan",bg='white')
en8.place(x=660, y=480)
lbl8 = Label(pro, text='Number of epochs', fg='black', bg='#c3ecf3', font=(NONE, 13, 'bold'))
lbl8.place(x=500, y=480)

lbl8 = Label(pro, text='Algorithm', fg='black', bg='#c3ecf3', font=(NONE, 13, 'bold'))
lbl8.place(x=120, y=590)

r1 = ttk.Radiobutton(pro, text='perceptron', value=1, variable=v)
r1.place(x=230, y=590)

r2 = ttk.Radiobutton(pro, text='Adaline', value=2, variable=v)
r2.place(x=330, y=590)

c=Checkbutton(pro,text='Add Bais',variable=g,onvalue=1,offvalue=0,font=(NONE, 13, 'bold'),padx=30,background="#c3ecf3")
c.place(x=500, y=590)

bt1 = Button(text='Done', fg='black', bg='#c3ecf3', width='25', height='2',font=('Helvetic', 12, 'italic', 'bold'),
             activebackground='black', activeforeground='white')
bt1.place(x=350, y=700)

# from PIL import Image, ImageTk
# b = Image.open("1.jfif")

# from tkinter import *
# from tkinter import messagebox
#
# top=Tk()
# top.geometry("800x700")
# c=Canvas(top,bg="gray16",height=200,width=200)
# filename=PhotoImage(file="VK.png")
# background_label=Label(top,image=filename)
# background_label.place(x=0,y=0,relwidth=1,relheight=1)
#
# c.pack()
# top.mainloop()
#
# # resized = b.resize((930,790))
#
pro.mainloop()
