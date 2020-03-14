# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:38:00 2020

@author: pc
"""

from tkinter import *

import tkinter.filedialog
from PIL import Image
from PIL import ImageTk
from PIL import ImageFilter
import matplotlib.pyplot as plt
from pylab import *

global image
import numpy as np
import scipy.signal
import cv2

fenetreP = Tk()
fenetreP.configure(background="#3781fb")
fenetreP.resizable(height=False, width=False)

#photo =PhotoImage(file = r"C:\Users\pc\IATI\ic.png")







gifdict = {}


def GaussianKernel(n, sigma):
    k = int(n / 2)
    h = np.zeros((2 * k + 1, 2 * k + 1))

    for i in range(-k, k + 1):

        for j in range(-k, k + 1):
            h[i + k][j + k] = (1 / (math.pi * 2 * sigma ** 2) * math.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2)))

    return h





def histogramme(im):
    histR = np.zeros((256))
    histG = np.zeros((256))
    histB = np.zeros((256))

    array = np.array(im)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            histR[array[i, j, 0]] += 1
            histG[array[i, j, 1]] += 1
            histB[array[i, j, 2]] += 1

    return histR, histG, histB


def histogrammeN(im):
    histN = np.zeros((256))
    

    array = np.array(im)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            histN[array[i, j]] += 1
            

    return histN

def open():
    c0.delete()
    filename = tkinter.filedialog.askopenfilename(title="Ouvrir une image", filetypes=[('Toutes les images', '*.*')])
    photo = ImageTk.PhotoImage(file=filename)
    

    gifdict[filename] = photo

    c0.create_image(0, 0, anchor=NW, image=photo)
    c0.config(height=photo.height(), width=photo.width())

    global image
    image = Image.open(filename)
    #hr, hv, hb = histogramme(image)
    hn=histogrammeN(image)
    figure()
    plt.plot(hn)
    
    plt.savefig("hist2.jpg")
    im3 = Image.open("hist2.jpg")
    size = (im3.size[0] / 2, im3.size[1] / 2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah"] = photo

    c2.create_image(0, 0, anchor=NW, image=photo)
    c2.config(height=photo.height(), width=photo.width())

k=GaussianKernel(5, 0.3)
def convolution():
    c1.delete()
    global image
    try:
        x
    except NameError:
        print("Ajoutez une photo SVP")
    src=np.array(image)


    img = cv2.GaussianBlur(src, (5, 5), 0)
    img2 = Image.fromarray(img, 'RGB')


    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist5.jpg")
    im3 = Image.open("hist5.jpg")
    size = (im3.size[0] / 2, im3.size[1] / 2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah3"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())





def filtremoy():
    c1.delete()
    global image
    try:
        x
    except NameError:
        print("Ajoutez une photo SVP")
    im2 = image.filter(ImageFilter.Kernel((3, 3), [1, 1, 1, 1, 1, 1, 1, 1, 1], 9))
    photo = ImageTk.PhotoImage(im2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(im2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist3.jpg")
    im3 = Image.open("hist3.jpg")
    size = (im3.size[0] / 2, im3.size[1] / 2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah3"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())




def ega():
    c1.delete()
    global image
    try:
        x
    except NameError:
        print("Ajoutez une photo SVP")
    src = np.array(image)
    im = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


    im2 = cv2.equalizeHist(im)
    img2 = Image.fromarray(im2, mode='L')
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    imgg = img2[:, :, 0]
    hr, hv, hb = histogramme(imgg)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist3.jpg")
    im3 = Image.open("hist3.jpg")
    size = (im3.size[0] / 2, im3.size[1] / 2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah3"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())

def maxy():
    c1.delete()
    global image
    try:
        x
    except NameError:
        print("Ajoutez une photo SVP")
    im2 = image.filter(ImageFilter.MaxFilter)
    photo = ImageTk.PhotoImage(im2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(im2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist6.jpg")
    im3 = Image.open("hist6.jpg")
    size = (im3.size[0] / 2, im3.size[1] / 2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah3"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())



def min():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    im2 = image.filter(ImageFilter.MinFilter)
    photo = ImageTk.PhotoImage(im2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(im2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist6.jpg")
    im3 = Image.open("hist6.jpg")
    size = (im3.size[0] / 2, im3.size[1] / 2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah3"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())



def mediane():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    img=np.array(image)
    im2 =cv2.medianBlur(img,5)
    img2 = Image.fromarray(im2, 'RGB')

    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(im2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist3.jpg")
    im3 = Image.open("hist3.jpg")
    size = (im3.size[0] / 2, im3.size[1] / 2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah3"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())


def filtremoy5():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    im2 = image.filter(
        ImageFilter.Kernel((5, 5), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 25))
    photo = ImageTk.PhotoImage(im2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(im2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
def Dil1():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")    
    m=np.array(image)
    mc=np.zeros((m.shape[0],m.shape[1]))
    md=np.zeros((m.shape[0],m.shape[1]))
    for i in range(1,m.shape[0]-1):
        for j in range(1,m.shape[1]-1):
            if m[i,j-1]==255 or m[i-1,j]==255  :
                md[i,j]=255
    for i in range(1,m.shape[0]-1):
        for j in range(1,m.shape[1]-1):
            if md[i,j-1]==255 or md[i-1,j]==255  :
                mc[i,j]=255
    img2 = Image.fromarray(mc)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
def Dil1():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    mc=np.zeros((m.shape[0],m.shape[1]))
    md=np.zeros((m.shape[0],m.shape[1]))
    iteration=8
    
    for u in range(iteration):
        for i in range(1,m.shape[0]-1):
            for j in range(1,m.shape[1]-1):
                if m[i,j-1]==255 or m[i-1,j]==255  :
                    md[i,j]=255
    img2 = Image.fromarray(md)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    



def Dil2():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")    
    m=np.array(image)
    mc=np.zeros((m.shape[0],m.shape[1]))
    md=np.zeros((m.shape[0],m.shape[1]))
    for i in range(1,m.shape[0]-1):
        for j in range(1,m.shape[1]-1):
            if m[i,j-1]==255:
                md[i,j]=255
   
    img2 = Image.fromarray(md)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
def Dil3():
    c1.delete()
    global image   
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    mc=np.zeros((m.shape[0],m.shape[1]))
    md=np.zeros((m.shape[0],m.shape[1]))
    iteration=1
    for u in range(iteration):
        for i in range(1,m.shape[0]-1):
            for j in range(1,m.shape[1]-1):
                if m[i,j+1]==255:
                    md[i,j]=255
   
    img2 = Image.fromarray(md)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    


def Dil4():
    c1.delete()
    global image 
    try:
        x
    except NameError:
        print("Ajoutez une photo SVP")
 
    m=np.array(image)
    mc=np.zeros((m.shape[0],m.shape[1]))
    md=m.copy()
    iteration=8
    
    for u in range(iteration):
        for i in range(1,m.shape[0]-1):
            for j in range(1,m.shape[1]-1):
                if m[i,j-1]==255 or m[i-1,j]==255 or  m[i+1,j]==255 or m[i,j+1]==255 :
                    md[i,j]=255
    img2 = Image.fromarray(md)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
    
    
def Ero4():
    c1.delete()
    global image 
    try:
        x
    except NameError:
        print("Ajoutez une photo SVP")
 
    m=np.array(image)
    mc=np.zeros((m.shape[0],m.shape[1]))
    md=m.copy()
    iteration=1
    
    for u in range(iteration):
        for i in range(1,m.shape[0]-1):
            for j in range(1,m.shape[1]-1):
                if m[i,j-1]==255 or m[i-1,j]==255 or  m[i+1,j]==255 or m[i,j+1]==255 :
                    md[i,j]=0
    img2 = Image.fromarray(md)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
def Ero1():
    c1.delete()
    global image 
    try:
        x
    except NameError:
        print("Ajoutez une photo SVP")
 
    m=np.array(image)
    mc=np.zeros((m.shape[0],m.shape[1]))
    md=m.copy()
    iteration=1
    
    for u in range(iteration):
        for i in range(1,m.shape[0]-1):
            for j in range(1,m.shape[1]-1):
                if m[i,j-1]==255 or m[i-1,j]==255:
                    md[i,j]=0
    img2 = Image.fromarray(md)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
    
def Ero3():
    c1.delete()
    global image 
    try:
        x
    except NameError:
        print("Ajoutez une photo SVP")
 
    m=np.array(image)
    mc=np.zeros((m.shape[0],m.shape[1]))
    md=m.copy()
    iteration=7
    
    for u in range(iteration):
        for i in range(1,m.shape[0]-1):
            for j in range(1,m.shape[1]-1):
                if m[i,j-1]==0 or m[i,j+1]==0:
                    md[i,j]=0
    img2 = Image.fromarray(md)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
def Ero2():
    c1.delete()
    global image 
    try:
        x
    except NameError:
        print("Ajoutez une photo SVP")
 
    m=np.array(image)
    mc=np.zeros((m.shape[0],m.shape[1]))
    md=m.copy()
    iteration=4
    
    for u in range(iteration):
        for i in range(1,m.shape[0]-1):
            for j in range(1,m.shape[1]-1):
                if m[i,j]==0:
                    md[i+1,j]=0
    img2 = Image.fromarray(md)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
def gradient1():
    c1.delete()
    global image
    
    m=np.array(image)
    gra = np.array((
	[-1,-1,-1],
	[-1,8,-1],
	[-1,-1,-1]), dtype="int")
    im= cv2.filter2D(m,-1,gra)
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
    

def laplacian():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")
    im= cv2.filter2D(m,-1,laplacian)
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
def Delatation_pre():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    kernel = np.ones((3,3),np.uint8)
    im = cv2.dilate(m,kernel,iterations = 1)
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
def Contour():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    kernel = np.ones((3,3),np.uint8)
    im = cv2.dilate(m,kernel,iterations = 2)
    im44 = cv2.erode(m,kernel,iterations = 1)
    img2 = Image.fromarray(im-im44)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())


def Ero_pre():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    kernel = np.ones((3,3),np.uint8)
    im = cv2.erode(m,kernel,iterations = 1)
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
    
def RobinsonY():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    RobinsonY = np.array((
	[-1,1,1],
	[-1,-2,1],
	[-1,1,1]), dtype="int")
    im= cv2.filter2D(m,-1,RobinsonY)
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
    
def RobinsonX():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    RobinsonX = np.array((
	[-1,-1,-1],
	[1,-2,1],
	[1,1,1]), dtype="int")
    im= cv2.filter2D(m,-1,RobinsonX)
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
def Robinsonsum():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    RobinsonX = np.array((
	[-1,-1,-1],
	[1,-2,1],
	[1,1,1]), dtype="int")
    RobinsonY = np.array((
	[-1,1,1],
	[-1,-2,1],
	[-1,1,1]), dtype="int")
    im4= cv2.filter2D(m,-1,RobinsonY)
    im= cv2.filter2D(m,-1,RobinsonX)
    img2 = Image.fromarray(im+im4)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
def Robinson45():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    Robinson45 = np.array((
	[1,1,1],
	[-1,-2,1],
	[-1,-1,1]), dtype="int")
    im= cv2.filter2D(m,-1,Robinson45)
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
def Robinson452():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    Robinson452 = np.array((
	[-1,-1,1],
	[-1,-2,1],
	[1,1,1]), dtype="int")
    im= cv2.filter2D(m,-1,Robinson452)
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
    
    

    
    













    
    
    
def kirchY():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    kirch = np.array((
	[-3,-3,5],
	[-3,0,5],
	[-3,-3,5]), dtype="int")
    im= cv2.filter2D(m,-1,kirch)
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
def kirchX():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    kirch = np.array((
	[-3,-3,-3],
	[-3,0,-3],
	[5,5,5]), dtype="int")
    im= cv2.filter2D(m,-1,kirch)
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())



def kirchsum():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    kirch1 = np.array((
	[-3,-3,5],
	[-3,0,5],
	[-3,-3,5]), dtype="int")
    kirch2 = np.array((
	[-3,-3,-3],
	[-3,0,-3],
	[5,5,5]), dtype="int")
    im= cv2.filter2D(m,-1,kirch1)
    imm= cv2.filter2D(m,-1,kirch2)
    img2 = Image.fromarray(im+imm)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
def prewit():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    prex = np.array((
	[-1,-1,-1],
	[0,0,0],
	[1,1,1]), dtype="int")
   
    im= cv2.filter2D(m,-1,prex)
    #im2= cv2.filter2D(m,-1,pre2)
    #img2 = Image.fromarray(im)
    #img2 = Image.fromarray(im)
    
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
def prewit45():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    prex = np.array((
	[0,1,1],
	[-1,0,1],
	[-1,-1,0]), dtype="int")
   
    im= cv2.filter2D(m,-1,prex)
    #im2= cv2.filter2D(m,-1,pre2)
    #img2 = Image.fromarray(im)
    #img2 = Image.fromarray(im)
    
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
def prewit452():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    prex = np.array((
	[-1,-1,0],
	[-1,0,1],
	[0,1,1]), dtype="int")
   
    im= cv2.filter2D(m,-1,prex)
    #im2= cv2.filter2D(m,-1,pre2)
    #img2 = Image.fromarray(im)
    #img2 = Image.fromarray(im)
    
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
def kirch45():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    prex = np.array((
	[-3,-3,-3],
	[5,0,-3],
	[0,5,-3]), dtype="int")
   
    im= cv2.filter2D(m,-1,prex)
    #im2= cv2.filter2D(m,-1,pre2)
    #img2 = Image.fromarray(im)
    #img2 = Image.fromarray(im)
    
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    
def kirch452():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    prex = np.array((
	[5,5,-3],
	[5,0,-3],
	[-3,-3,-3]), dtype="int")
   
    im= cv2.filter2D(m,-1,prex)
    #im2= cv2.filter2D(m,-1,pre2)
    #img2 = Image.fromarray(im)
    #img2 = Image.fromarray(im)
    
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
    


def prewity():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    prex = np.array((
	[-1,-1,-1],
	[0,0,0],
	[1,1,1]), dtype="int")
    prey = np.array((
	[-1,0,1],
	[-1,0,1],
	[-1,0,1]), dtype="int")
    im= cv2.filter2D(m,-1,prey)
    #im2= cv2.filter2D(m,-1,pre2)
    #img2 = Image.fromarray(im)
    #img2 = Image.fromarray(im)
    
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())




def prewitsum():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    m=np.array(image)
    prex = np.array((
	[-1,-1,-1],
	[0,0,0],
	[1,1,1]), dtype="int")
    prey = np.array((
	[-1,0,1],
	[-1,0,1],
	[-1,0,1]), dtype="int")
    iss= cv2.filter2D(m,-1,prex)
    im= cv2.filter2D(m,-1,prey)
    #im2= cv2.filter2D(m,-1,pre2)
    #img2 = Image.fromarray(im)
    #img2 = Image.fromarray(im)
    
    img2 = Image.fromarray(iss+im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
def sobel():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    
    m=np.array(image)
    sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")


    sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")
    im= cv2.filter2D(m,-1,sobelX)
    dst2 = cv2.filter2D(m,-1,sobelY)
  
    img2 = Image.fromarray(im+dst2)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())
    
def sobelX():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    
    m=np.array(image)
    sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")


    im= cv2.filter2D(m,-1,sobelX)
    
  
    img2 = Image.fromarray(im)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())



def sobelY():
    c1.delete()
    global image
    try:
        image
    except NameError:
        print("Ajoutez une photo SVP")
    
    m=np.array(image)
    


    sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")
    
    dst2 = cv2.filter2D(m,-1,sobelY)
  
    img2 = Image.fromarray(dst2)
    
    photo = ImageTk.PhotoImage(img2)
    gifdict["ima"] = photo
    c1.create_image(0, 0, anchor=NW, image=photo)
    c1.config(height=photo.height(), width=photo.width())
    hr, hv, hb = histogramme(img2)
    figure()
    plt.plot(hr)
    plt.plot(hv)
    plt.plot(hb)
    plt.savefig("hist4.jpg")
    im3 = Image.open("hist4.jpg")
    size = (im3.size[0]/2,  im3.size[1]/2)
    im3.thumbnail(size)
    photo = ImageTk.PhotoImage(im3)
    gifdict["imah4"] = photo
    c3.create_image(0, 0, anchor=NW, image=photo)
    c3.config(height=photo.height(), width=photo.width())









fenetreP.title("IATI UI")
#fenetreP.configure(ico)
menubar = Menu(fenetreP)
menuFichier = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Fichier", menu=menuFichier)
menuFichier.add_cascade(label="Open", command=open)
menuFichier.add_cascade(label="Exit", command=fenetreP.destroy)
fenetreP.config(menu=menubar)
menuLiniaire = Menu(menubar)
menubar.add_cascade(label="Filtrage Liniaire", menu=menuLiniaire)
menuLiniaire.add_command(label="Moyenneur 3x3", command=filtremoy)
menuLiniaire.add_command(label="Moyenneur 5x5", command=filtremoy5)
menuLiniaire.add_command(label="Gaussien", command=convolution)

menuNonLiniaire = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Filtrage non Liniaire", menu=menuNonLiniaire)
menuNonLiniaire.add_command(label="Median", command=mediane)
menuNonLiniaire.add_command(label="Min", command=min)
menuNonLiniaire.add_command(label="Max", command=maxy)

menuSeuillage = Menu(menubar, tearoff=0)
menubar.add_cascade(label="seuillage", menu=menuSeuillage)
menuSeuillage.add_command(label="median", command=None)
menuSeuillage.add_command(label="Moyen", command=None)
menuSeuillage.add_command(label="Milieu", command=None)
menuSeuillage.add_command(label="Egalisation", command=ega)
menuG = Menu(menubar, tearoff=0)
menuG.add_command(label="laplacian", command=laplacian)




menuSobel = Menu(menubar, tearoff=0)
menuSobel.add_command(label="Sobel X", command=sobelX)
menuSobel.add_command(label="Sobel Y", command=sobelY)
menuSobel.add_command(label="Sobel", command=sobel)


menuPrewit = Menu(menubar, tearoff=0)
menuPrewit.add_command(label="Prewitt x", command=prewit)
menuPrewit.add_command(label="Prewitt y", command=prewity)
menuPrewit.add_command(label="Prewitt diagonale 1", command=prewit45)
menuPrewit.add_command(label="Prewitt diagonale 2", command=prewit452)
menuPrewit.add_command(label="Prewitt", command=prewitsum)


menuG.add_cascade(label="sobel", menu=menuSobel)
menuG.add_cascade(label="Prewitt", menu=menuPrewit)


menuKirch = Menu(menubar, tearoff=0)
menuKirch.add_command(label="Kirch X", command=kirchX)
menuKirch.add_command(label="Kirch Y", command=kirchY)
menuKirch.add_command(label="kirch diagonale 1", command=kirch45)
menuKirch.add_command(label="kirch diagonale 2", command=kirch452)
menuKirch.add_command(label="Kirch", command=kirchsum)

menuRobinson = Menu(menubar, tearoff=0)
menuRobinson.add_command(label="Robinson X", command=RobinsonX)
menuRobinson.add_command(label="Robinson Y", command=RobinsonY)
menuRobinson.add_command(label="Robinson diagonale 1", command=Robinson45)
menuRobinson.add_command(label="Robinson diagonale 2", command=Robinson452)
menuRobinson.add_command(label="Robinson", command=Robinsonsum)





menuSegmentation = Menu(menubar, tearoff=0)
menuMandil = Menu(menubar, tearoff=0)
menuPredil = Menu(menubar, tearoff=0)
menuManero = Menu(menubar, tearoff=0)
menuPreero = Menu(menubar, tearoff=0)
menubar.add_cascade(label="segmentation", menu=menuSegmentation)
menuSegmentation.add_cascade(label="gardient",menu=menuG)
menuMandil.add_command(label="dilatation L", command=Dil1)
menuMandil.add_command(label="dilatation 2", command=Dil2)
menuMandil.add_command(label="dilatation 3", command=Dil3)
menuMandil.add_command(label="dilatation +", command=Dil4)


menuManero.add_command(label="erosion +", command=Ero4)
menuManero.add_command(label="erosion I", command=Ero2)
menuManero.add_command(label="erosion L", command=Ero1)
menuManero.add_command(label="contour", command=Contour)



menuG.add_cascade(label="Kirch", menu=menuKirch)
menuG.add_cascade(label="Robinson", menu=menuRobinson)


menuDil= Menu(menubar, tearoff=0)
mmm= Menu(menubar, tearoff=0)
menuDil.add_cascade(label="Manuel",menu=menuMandil)
menuDil.add_command(label="Predefinie",command=Delatation_pre)
mmm.add_cascade(label="Manuel",menu=menuManero)
mmm.add_command(label="Predefinie",command=Ero_pre)
menuEro= Menu(menubar, tearoff=0)
menuMor= Menu(menubar, tearoff=0)

menuMor.add_cascade(label="Dilatation",menu=menuDil)
menuMor.add_cascade(label="Erosion",menu=mmm)

menuSegmentation.add_cascade(label="Morphologique", menu=menuMor)





c0 = Canvas(fenetreP)
c0.grid(row=0, column=0)
c1 = Canvas(fenetreP)
c1.grid(row=0, column=1)
c2 = Canvas(fenetreP)
c2.grid(row=1, column=0)
c3 = Canvas(fenetreP)
c3.grid(row=1, column=1)
#b2 = Button(fenetreP, text = "GFG") 
#b2.grid(row=1, column=1) 
fenetreP.mainloop()
