from tkinter import *
from PIL import ImageTk,Image

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import datasets,metrics
import sklearn.datasets as datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
from sklearn import tree
from imblearn.over_sampling import RandomOverSampler


def getvals():
    Temperature_en=Temperature_val.get()
    Dew_en=Dew_val.get()
    Hunidity_en=Hunidity_val.get()
    Wind_en=Win_val.get()
    Visibility_en=Visibility_val.get()
    Precipitation_en=Pressure_val.get()

    D=DT.predict([[Temperature_en, Dew_en, Hunidity_en, Wind_en,Visibility_en,Precipitation_en]]) 
    
    str=""
    drizzle_window = Toplevel(root)
    drizzle_window.iconbitmap('favicon.ico')
    drizzle_window.configure(bg="#4D99E7")       
    drizzle_window.title("   PREDICTION")
    drizzle_window.geometry("600x250")
    drizzle_window.minsize(600,250)
    drizzle_window.maxsize(600,250)
    Label(drizzle_window, text="Weather will be :", font="comicsansms 13 bold",bg="#4D99E7", pady=2).place(x=15, y=45)
    Label(drizzle_window, text=str.join(D), font="comicsansms 13",bg="#4D99E7", pady=2).place(x=155, y=45)
    
    mainloop()
  
def visual():
    sns.pairplot(dataset, hue='Weather')
    plt.show()
    
def cor():
    fig,ax=plt.subplots(1,1,figsize=(18,10))
    sns.heatmap(numeric_dataset.corr(method ='kendall'),annot=True,linewidths=1.5)
    plt.show()
    
def tree_v():
    plt.figure(figsize=(30,30))
    tree.plot_tree(DT,filled=True)
    plt.show()
def accurecy():
        drizzle_window = Toplevel(root)
        drizzle_window.iconbitmap('favicon.ico')
        drizzle_window.configure(bg="#4D99E7")
        drizzle_window.title("  TRAINING ACCURECY")
        drizzle_window.geometry("1300x800")
        
        Label(drizzle_window, text="ACCURECY OF THE TRAINING IS :", font="comicsansms 13 bold",bg="#4D99E7", pady=2).place(x=25, y=45)
        Label(drizzle_window, text=classification_report(Y_test,Y_pred), font="comicsansms 13",bg="#4D99E7", pady=2).place(x=300, y=45)

def data_details():
        drizzle_window = Toplevel(root)
        drizzle_window.iconbitmap('favicon.ico')
        drizzle_window.configure(bg="#4D99E7")
        drizzle_window.title("  DATASET USED")
        drizzle_window.geometry("950x600")
        drizzle_window.minsize(950,600)
        drizzle_window.maxsize(950,600)
        Label(drizzle_window, text="DETAILES OF DATASET :", font="comicsansms 13 bold",bg="#4D99E7", pady=2).place(x=10, y=10)
        label = Label(drizzle_window, text=dataset, font="comicsansms 13",bg="#4D99E7", pady=2)
        label.place(x=110, y=40)
        Label(drizzle_window, text="DATA TYPES :", font="comicsansms 13 bold",bg="#4D99E7", pady=2).place(x=10, y=320)
        Label(drizzle_window, text=dataset['Temp_C'].dtype, font="comicsansms 13",bg="#4D99E7", pady=2).place(x=110, y=350)
        Label(drizzle_window, text=dataset['Dew Point Temp_C'].dtype, font="comicsansms 13",bg="#4D99E7", pady=2).place(x=110, y=380)
        Label(drizzle_window, text=dataset['Rel Hum_%'].dtype, font="comicsansms 13",bg="#4D99E7", pady=2).place(x=110, y=410)
        Label(drizzle_window, text=dataset['Wind Speed_km/h'].dtype, font="comicsansms 13",bg="#4D99E7", pady=2).place(x=110, y=440)
        Label(drizzle_window, text=dataset['Visibility_km'].dtype, font="comicsansms 13",bg="#4D99E7", pady=2).place(x=110, y=470)
        Label(drizzle_window, text=dataset['Press_kPa'].dtype, font="comicsansms 13",bg="#4D99E7", pady=2).place(x=110, y=500)
        Label(drizzle_window, text=dataset['Weather'].dtype, font="comicsansms 13",bg="#4D99E7", pady=2).place(x=110, y=530)
        mainloop()    

def matrix():
    class_names = ['Fog', 'Freezing Drizzle,Fog', 'Mostly Cloudy', 'Cloudy', 'Rain',
       'Rain Showers', 'Mainly Clear', 'Snow Showers', 'Snow', 'Clear',
       'Freezing Rain,Fog', 'Freezing Rain', 'Freezing Drizzle',
       'Rain,Snow', 'Moderate Snow', 'Freezing Drizzle,Snow',
       'Freezing Rain,Snow Grains', 'Snow,Blowing Snow', 'Freezing Fog',
       'Haze', 'Rain,Fog', 'Drizzle,Fog', 'Drizzle',
       'Freezing Drizzle,Haze', 'Freezing Rain,Haze', 'Snow,Haze',
       'Snow,Fog', 'Snow,Ice Pellets', 'Rain,Haze', 'Thunderstorms,Rain',
       'Thunderstorms,Rain Showers', 'Thunderstorms,Heavy Rain Showers',
       'Thunderstorms,Rain Showers,Fog', 'Thunderstorms',
       'Thunderstorms,Rain,Fog',
       'Thunderstorms,Moderate Rain Showers,Fog', 'Rain Showers,Fog',
       'Rain Showers,Snow Showers', 'Snow Pellets', 'Rain,Snow,Fog',
       'Moderate Rain,Fog', 'Freezing Rain,Ice Pellets,Fog',
       'Drizzle,Ice Pellets,Fog', 'Drizzle,Snow', 'Rain,Ice Pellets',
       'Drizzle,Snow,Fog', 'Rain,Snow Grains', 'Rain,Snow,Ice Pellets',
       'Snow Showers,Fog', 'Moderate Snow,Blowing Snow']  
    plt.imshow(cm, interpolation='none', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()


    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=270)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

root= Tk()
dataset = pd.read_csv('seattle-weather.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 6:7].values
sm=RandomOverSampler()
X,Y=sm.fit_resample(X,Y)
numeric_dataset = dataset.select_dtypes(include=[np.number])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.99, random_state=4)

DT = tree.DecisionTreeClassifier()
DT = DT.fit(X_train, Y_train)
Y_pred=DT.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)

root.title("WEATHER")
root.geometry("975x650")
root.minsize(975,650)
root.maxsize(975,650)

image = Image.open("25502.jpg")
photo=ImageTk.PhotoImage(image)
back_photo=Label(image=photo)
back_photo.place(x=0,y=0)
root.iconbitmap('favicon.ico')

Label(root,text="Temperature(C):",font="comicsansms 13",bg="#4D99E7").place(x=165,y=50)
Label(root,text="Dew Point Temp(C):",font="comicsansms 13",bg="#4D99E7").place(x=165,y=94)
Label(root,text="Hunidity(%):",font="comicsansms 13",bg="#4D99E7").place(x=165,y=138)
Label(root,text="Wind Speed(km/h):",font="comicsansms 13",bg="#4D99E7").place(x=165,y=182)
Label(root,text="Visibility(km):",font="comicsansms 13",bg="#4D99E7").place(x=165,y=226)
Label(root,text="Pressure(kPa):",font="comicsansms 13",bg="#4D99E7").place(x=165,y=270)

Temperature_val=DoubleVar()
Dew_val=DoubleVar()
Hunidity_val=DoubleVar()
Win_val=DoubleVar()
Visibility_val=DoubleVar()
Pressure_val=DoubleVar()

Temperature_en=Entry(root,textvariable=Temperature_val,).place(x=370,y=50)
Dew_en=Entry(root,textvariable=Dew_val).place(x=370,y=94)
Hunidity_en=Entry(root,textvariable=Hunidity_val).place(x=370,y=138)
Wind_en=Entry(root,textvariable=Win_val).place(x=370,y=182)
Visibility_en=Entry(root,textvariable=Visibility_val).place(x=370,y=226)
Pressure_en=Entry(root,textvariable=Pressure_val).place(x=370,y=270)

Button(text="CHECK",command=getvals,bg="green",pady=6,padx=10).place(x=370,y=320)
Button(text="DATA VISUALIZATION",command=visual,bg="green",pady=6,padx=10).place(x=800,y=50)
Button(text="CONFUSION MATRIX",command=matrix,bg="green",pady=6,padx=12).place(x=800,y=94)
Button(text="TREE VISUALIZATION",command=tree_v,bg="green",pady=6,padx=13).place(x=800,y=138)
Button(text="DATA DETAILS",command=data_details,bg="green",pady=6,padx=30).place(x=800,y=182)
Button(text="TRAINING ACCURECY",command=accurecy,bg="green",pady=6,padx=10).place(x=800,y=226)
Button(text="DATA CORRELATION",command=cor,bg="green",pady=6,padx=10).place(x=800,y=270)

root.mainloop()