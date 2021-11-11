# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:20:16 2021

@author: pc
"""



import cv2
import timeit
from PyQt5.QtWidgets import QFileDialog
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from PyQt5 import QtCore, QtGui, QtWidgets




#changer alphae pour un taux plus et precision
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        #*********Button**
        
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(900, 300, 181, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setStyleSheet("QPushButton"
                                      "{"
                                      "background-color : lightblue;"
                                      "}"
                                      "QPushButton::pressed"
                                      "{"
                                      "background-color:white;"
                                      "}"
                                      )
        
        
        
        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setGeometry(QtCore.QRect(900, 550, 181, 31))
        self.pushButton2.setObjectName("pushButton")
        self.pushButton2.setStyleSheet("QPushButton"
                                      "{"
                                      "background-color : lightblue;"
                                      "}"
                                      "QPushButton::pressed"
                                      "{"
                                      "background-color:white;"
                                      "}"
                                      )
        self.pushButton3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton3.setGeometry(QtCore.QRect(900, 600, 181, 31))
        self.pushButton3.setObjectName("pushButton")
        self.pushButton3.setStyleSheet("QPushButton"
                                      "{"
                                      "background-color : lightblue;"
                                      "}"
                                      "QPushButton::pressed"
                                      "{"
                                      "background-color:white;"
                                      "}"
                                      )
        self.pushButton4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton4.setGeometry(QtCore.QRect(900, 650, 181, 31))
        self.pushButton4.setObjectName("pushButton")
        self.pushButton4.setStyleSheet("QPushButton"
                                      "{"
                                      "background-color : lightblue;"
                                      "}"
                                      "QPushButton::pressed"
                                      "{"
                                      "background-color:white;"
                                      "}"
                                      )
       
        #****** label *****
        self.ImageOrigine = QtWidgets.QLabel(self.centralwidget)
        self.ImageOrigine.setGeometry(QtCore.QRect(410, 300, 351, 231))
        self.ImageOrigine.setText("")
        self.ImageOrigine.setObjectName("label_3")
        
        self.ImageOrigine.setStyleSheet("background-color:lightblue;")
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(450, 50, 911, 21))
        font=QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(100, 500, 551, 231))
        self.label_7.setText("Veuillez choisir votre image")
        self.label_7.setObjectName("label_7")
        font=QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label")
        
        
        #***label table***
        self.labe1 = QtWidgets.QLabel(self.centralwidget)
        self.labe1.setGeometry(QtCore.QRect(50, 80, 140, 51))
        self.labe1.setStyleSheet("border: 1px solid black;padding:10px")
        font=QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        font.setWeight(75)
        self.labe1.setFont(font)
        self.labe1.setObjectName("label")
        
        self.labe2 = QtWidgets.QLabel(self.centralwidget)
        self.labe2.setGeometry(QtCore.QRect(190, 80, 210, 51))
        self.labe2.setStyleSheet("border: 1px solid black;padding:10px")
        font=QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        font.setWeight(75)
        self.labe2.setFont(font)
        self.labe2.setObjectName("label")
        
        self.labe3 = QtWidgets.QLabel(self.centralwidget)
        self.labe3.setGeometry(QtCore.QRect(400, 80, 300, 51))
        self.labe3.setStyleSheet("border: 1px solid black;padding:10px")
        font=QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        font.setWeight(75)
        self.labe3.setFont(font)
        self.labe3.setObjectName("label")
        
        self.labe4 = QtWidgets.QLabel(self.centralwidget)
        self.labe4.setGeometry(QtCore.QRect(700, 80, 300, 51))
        self.labe4.setStyleSheet("border: 1px solid black;padding:10px")
        font=QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        font.setWeight(75)
        self.labe4.setFont(font)
        self.labe4.setObjectName("label")
        
        self.labe5 = QtWidgets.QLabel(self.centralwidget)
        self.labe5.setGeometry(QtCore.QRect(50, 130, 140, 51))
        self.labe5.setStyleSheet("border: 1px solid black; padding:10px")
        font=QtGui.QFont()
        font.setPointSize(10)
        font.setItalic(True)
        font.setWeight(75)
        self.labe5.setFont(font)
        self.labe5.setObjectName("label")
        
        self.labe6 = QtWidgets.QLabel(self.centralwidget)
        self.labe6.setGeometry(QtCore.QRect(190, 130, 210, 51))
        self.labe6.setStyleSheet("border: 1px solid black;")
        font=QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        font.setWeight(45)
        self.labe6.setFont(font)
        self.labe6.setObjectName("label")
        
        self.labe7 = QtWidgets.QLabel(self.centralwidget)
        self.labe7.setGeometry(QtCore.QRect(400,130, 300, 51))
        self.labe7.setStyleSheet("border: 1px solid black;")
        font=QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        font.setWeight(45)
        self.labe7.setFont(font)
        self.labe7.setObjectName("label")
        
        self.labe8 = QtWidgets.QLabel(self.centralwidget)
        self.labe8.setGeometry(QtCore.QRect(700, 130, 300, 51))
        self.labe8.setStyleSheet("border: 1px solid black;")
        font=QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        font.setWeight(45)
        self.labe8.setFont(font)
        self.labe8.setObjectName("label")
        
        self.labe9 = QtWidgets.QLabel(self.centralwidget)
        self.labe9.setGeometry(QtCore.QRect(50, 180, 140, 51))
        self.labe9.setStyleSheet("border: 1px solid black;padding:10px")
        font=QtGui.QFont()
        font.setPointSize(10)
        font.setItalic(True)
        font.setWeight(75)
        self.labe9.setFont(font)
        self.labe9.setObjectName("label")
        
        self.labe10= QtWidgets.QLabel(self.centralwidget)
        self.labe10.setGeometry(QtCore.QRect(190, 180, 210, 51))
        self.labe10.setStyleSheet("border: 1px solid black;")
        font=QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        font.setWeight(45)
        self.labe10.setFont(font)
        self.labe10.setObjectName("label")
        
        self.labe11 = QtWidgets.QLabel(self.centralwidget)
        self.labe11.setGeometry(QtCore.QRect(400, 180, 300, 51))
        self.labe11.setStyleSheet("border: 1px solid black;")
        font=QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        font.setWeight(45)
        self.labe11.setFont(font)
        self.labe11.setObjectName("label")
        
        self.labe12 = QtWidgets.QLabel(self.centralwidget)
        self.labe12.setGeometry(QtCore.QRect(700, 180, 300, 51))
        self.labe12.setStyleSheet("border: 1px solid black;")
        font=QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        font.setWeight(45)
        self.labe12.setFont(font)
        self.labe12.setObjectName("label")
        
        self.labe13 = QtWidgets.QLabel(self.centralwidget)
        self.labe13.setStyleSheet("border: 1px solid black;padding:10px")
        self.labe13.setGeometry(QtCore.QRect(50, 230, 140, 51))
        font=QtGui.QFont()
        font.setPointSize(10)
        font.setItalic(True)
        font.setWeight(75)
        self.labe13.setFont(font)
        self.labe13.setObjectName("label")
        
        self.labe14 = QtWidgets.QLabel(self.centralwidget)
        self.labe14.setGeometry(QtCore.QRect(190, 230, 210, 51))
        self.labe14.setStyleSheet("border: 1px solid black;")
        font=QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        font.setWeight(45)
        self.labe14.setFont(font)
        self.labe14.setObjectName("label")
        
        self.labe15 = QtWidgets.QLabel(self.centralwidget)
        self.labe15.setGeometry(QtCore.QRect(400, 230, 300, 51))
        self.labe15.setStyleSheet("border: 1px solid black;")
        font=QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        font.setWeight(45)
        self.labe15.setFont(font)
        self.labe15.setObjectName("label")
        
        self.labe16 = QtWidgets.QLabel(self.centralwidget)
        self.labe16.setGeometry(QtCore.QRect(700, 230, 300, 51))
        self.labe16.setStyleSheet("border: 1px solid black;")
        font=QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(True)
        font.setWeight(45)
        self.labe16.setFont(font)
        self.labe16.setObjectName("label")
        
        
        
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        #*****line for sperate **
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(100, 250, 800, 601))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        
       
      
       
        
       
        
       
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Reconnaissance des Images"))
               
        self.pushButton.setText(_translate("MainWindow", "Ouvrir"))
        self.pushButton2.setText(_translate("MainWindow", "DecisionTreeClassifier"))
        self.pushButton3.setText(_translate("MainWindow", "Multi-Coches"))
        self.pushButton4.setText(_translate("MainWindow", "Multinomiale"))
        
       
        self.label.setText(_translate("MainWindow", "Comparaison Bayes Naîf:"))
        
        
        self.labe1.setText(_translate("MainWindow", "Classifier"))
        self.labe2.setText(_translate("MainWindow", "Temps d'execution"))
        self.labe3.setText(_translate("MainWindow", "Taux de reconnaissance %"))
        self.labe4.setText(_translate("MainWindow", "Taux de réussi par test %"))
        self.labe5.setText(_translate("MainWindow", "DTC"))
        
        #les calcules afficherr..................
        self.labe6.setText(_translate("MainWindow", str(t_dtc.timeit(1))))
        self.labe7.setText(_translate("MainWindow", str(dtc.score(d,alphabet)*100)))
        self.labe8.setText(_translate("MainWindow", str(dtc.score(d_test,alphabet_test)*100)))
        self.labe9.setText(_translate("MainWindow", "Multi-Couches"))
        self.labe10.setText(_translate("MainWindow",str(t_mcouche.timeit(1))))
        self.labe11.setText(_translate("MainWindow", str(mcouche.score(d,alphabet)*100)))
        self.labe12.setText(_translate("MainWindow", str(mcouche.score(d_test,alphabet_test)*100)))
        self.labe13.setText(_translate("MainWindow", "Multinomiale"))
        self.labe14.setText(_translate("MainWindow", str(t_m.timeit(1))))
        self.labe15.setText(_translate("MainWindow", str(multinm.score(d,alphabet)*100)))
        self.labe16.setText(_translate("MainWindow", str(multinm.score(d_test,alphabet_test)*100)))
        
        
        
        self.pushButton.clicked.connect(self.openFile)
        self.pushButton2.clicked.connect(self.DecisionTreeClassifier)
        self.pushButton3.clicked.connect(self.multicouche)
        self.pushButton4.clicked.connect(self.multinm)
        
       
       
      
    def openFile(self):
        nom_fichier = QFileDialog.getOpenFileName(None, 'Open file', '', "Image files (*.BMP *.jpg *.gif *.png)")
        self.path = nom_fichier[0]
        pathx = self.path
        pixmap = QtGui.QPixmap(pathx)
        img = cv2.imread(self.path,0)
          
        th2 = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        self.Xi=th2.flatten()
        self.ImageOrigine.setPixmap(pixmap)
        self.ImageOrigine.setScaledContents(1)
     
    #les resultats des classifieurs 
    def DecisionTreeClassifier(self):
      self.label_7.setText("le resultat DecisionTreeClassifier: =====> "+str(dtc.predict([self.Xi])))
       
    def multinm(self):
        self.label_7.setText("le resultat Multinomiale : =====> "+str(multinm.predict([self.Xi])))  
        
    def multicouche(self):
           self.label_7.setText("le resultat Multi-Couches : =====> "+str(mcouche.predict([self.Xi])))


#l'apprentissage des destributeurs
def DecisionTreeClassifier(d,alphabet):
       global dtc
     
       dtc = tree.DecisionTreeClassifier(ccp_alpha=0.005,criterion='entropy',splitter='random',random_state=5)
       dtc.fit(d,alphabet)
       return dtc
   
def multinomial(d,alphabet):
       global multinm
       multinm = MultinomialNB(alpha=0.5)
       multinm.fit(d,alphabet)   
       return multinm
    
def multicouche(d,alphabet):
      global mcouche
      mcouche=MLPClassifier(max_iter=1700,hidden_layer_sizes=(200,50),alpha=0.09)
      mcouche.fit(d,alphabet)
      return mcouche



#prétraitements des image de l'apprentissage 
def image_trait_app():
       d=[]
       for i in range(1,63):
            img = "./images/apprentissage/".__add__(i.__str__()).__add__(".png")
            img = cv2.imread(img,0)
            th2 = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    
            
            Xi=th2.flatten()
            d.append(Xi)
            
       return  d 
      
#temps d'execution de l'apprentissage
      
def temp_Apprendre()  : 


    
       temp_dtc = timeit.Timer('DecisionTreeClassifier(d,alphabet)','from __main__ import DecisionTreeClassifier, d, alphabet')

       temp_multi = timeit.Timer('multinomial(d,alphabet)','from __main__ import multinomial, d, alphabet')
        
       temp_mcoche=timeit.Timer('multicouche(d,alphabet)','from __main__ import multicouche, d, alphabet')
       
       return temp_dtc,temp_multi,temp_mcoche
       
       
#prétraitement des images de test
def image_trait_test():
     d=[]

     
     for i in range(1,42):
            img = "./images/TEST/".__add__(i.__str__()).__add__(".png")
            img = cv2.imread(img,0)
          
            th2 = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
     
            
            Xi=th2.flatten()
            d.append(Xi)
       
     return d
    

        
    
    
if __name__ == "__main__":
    import sys
#etiquettage des images de l'apprentissage et de test
    alphabet = ["A","A","A","A","A", "B","B","B", "C","C", "D","D", "E","E","E", "F","F", "G","G", "H","H", "I","I","I","I","I", "J","J", "K", "K", "L", "L", "M", "M", "N","N", "O","O","O", "P","P", "Q","Q", "R","R", "S","S", "T","T",
                 "U","U","U", "V","V", "W","W", "X","X", "Y","Y", "Z","Z"]
    
    alphabet_test = ["U","A","A","B","B","C","C","D","D","E","E","F","F","G","H","H","I","I","J","J","Z","K","Y","L","M","M","N","N", "O","O","P","P","Q", "X","W","R","S","S","V","T","U"]

#les images d'apprentissage et de test traité    
    d=image_trait_app()
    d_test=image_trait_test()
#calcule les temps d'apprentissage de déffirents destributeurs
    t_dtc,t_m,t_mcouche=temp_Apprendre()
    
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_()) 

