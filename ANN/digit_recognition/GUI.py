from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import tkinter as tk
import keras
import tensorflow as tf
import cv2
import numpy as np

window = Tk()
frame = Frame(
    window,
    bg="#a4b0be",
    width=200,
    height=200,
    bd=10,
    relief=RIDGE
)

lbImage = Label(frame)
lbResult = Label(window, text="Result: ", font=('Arial',20,'bold'))
lbPredict = Label(window, text="...", font=('Arial',20,'bold'))

# load model
model = keras.models.load_model('digitRecognition.keras')

def createWindow(width=480, height=360, title="Window") -> None:
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    
    window.geometry('%dx%d+%d+%d' % (width, height, x, y))
    window.minsize(width, height)
    window.maxsize(width, height)
    window.title(title)
    window.config(background="#ffffff")

def openFile():
    filePath = filedialog.askopenfilename(
        filetypes= [("Image file", "*.png *.jpg *.jpeg")]
    )
    if filePath:
        global srcImg
        srcImg = str(filePath)
        displayImage(filePath)
        btnPredict.config(state=ACTIVE)

def displayImage(filePath):
    lbImage.config(image='')
    image = Image.open(filePath)
    image_resize = image.resize((200,200))
    photo = ImageTk.PhotoImage(image=image_resize)

    lbImage.config(image=photo)
    lbImage.image = photo
    lbImage.pack()

def predict():
    lbPredict.config(text="")
    img = cv2.imread(srcImg, cv2.IMREAD_GRAYSCALE)

    rz_img = cv2.resize(img, dsize=(20,20))
    rz_img = rz_img.astype(np.float32) / 255
    rz_img = np.expand_dims(rz_img, axis=-1)
    rz_img = np.expand_dims(rz_img, axis=0)
    y_pred = model.predict(rz_img)
    result = str(np.argmax(y_pred.round(3)))
    lbPredict.config(text=result)

def main():
    createWindow(title="Digit Recognition")
    frame.place(x=50, y=50)

    icon = PhotoImage(file="img\\connection.png")
    window.iconphoto(True, icon)
    
    btnImportPhoto = Button(text="Import Photo", command=openFile, width=15, height=2, font=('Arial',10,'bold'), bg="#18dcff")
    btnImportPhoto.place(x=280, y=80)
    btnImportPhoto.config(activebackground="#45aaf2")

    global btnPredict
    btnPredict = Button(text="Predict", command=predict, width=15, height=2, font=('Arial',10,'bold'), bg="#32ff7e")
    btnPredict.place(x=280, y=170)
    btnPredict.config(state=DISABLED)
    btnPredict.config(activebackground="#7bed9f")

    lbResult.place(x=50,y=300)
    lbPredict.place(x=155,y=300)


    window.mainloop()

if __name__ == "__main__":
    main()