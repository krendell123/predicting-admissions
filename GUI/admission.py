# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'admission.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(679, 406)
        MainWindow.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.Title = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Lucida Grande")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.Title.setFont(font)
        self.Title.setAlignment(QtCore.Qt.AlignCenter)
        self.Title.setObjectName("Title")
        self.verticalLayout_3.addWidget(self.Title)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.triageLabel = QtWidgets.QLabel(self.centralwidget)
        self.triageLabel.setObjectName("triageLabel")
        self.horizontalLayout.addWidget(self.triageLabel)
        self.triageBox = QtWidgets.QComboBox(self.centralwidget)
        self.triageBox.setObjectName("triageBox")
        self.horizontalLayout.addWidget(self.triageBox)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.ambulanceLabel = QtWidgets.QLabel(self.centralwidget)
        self.ambulanceLabel.setObjectName("ambulanceLabel")
        self.horizontalLayout_2.addWidget(self.ambulanceLabel)
        self.ambulanceBox = QtWidgets.QComboBox(self.centralwidget)
        self.ambulanceBox.setObjectName("ambulanceBox")
        self.horizontalLayout_2.addWidget(self.ambulanceBox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.ageLabel = QtWidgets.QLabel(self.centralwidget)
        self.ageLabel.setObjectName("ageLabel")
        self.horizontalLayout_3.addWidget(self.ageLabel)
        self.ageBox = QtWidgets.QComboBox(self.centralwidget)
        self.ageBox.setObjectName("ageBox")
        self.horizontalLayout_3.addWidget(self.ageBox)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.prevAdmitLabel = QtWidgets.QLabel(self.centralwidget)
        self.prevAdmitLabel.setObjectName("prevAdmitLabel")
        self.horizontalLayout_4.addWidget(self.prevAdmitLabel)
        self.prevAdmitBox = QtWidgets.QComboBox(self.centralwidget)
        self.prevAdmitBox.setObjectName("prevAdmitBox")
        self.horizontalLayout_4.addWidget(self.prevAdmitBox)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_12.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.presentingLabel = QtWidgets.QLabel(self.centralwidget)
        self.presentingLabel.setObjectName("presentingLabel")
        self.horizontalLayout_6.addWidget(self.presentingLabel)
        self.presentingBox = QtWidgets.QComboBox(self.centralwidget)
        self.presentingBox.setObjectName("presentingBox")
        self.horizontalLayout_6.addWidget(self.presentingBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_2.addWidget(self.listWidget)
        self.horizontalLayout_12.addLayout(self.verticalLayout_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_12)
        self.calcBtn = QtWidgets.QPushButton(self.centralwidget)
        self.calcBtn.setObjectName("calcBtn")
        self.verticalLayout_3.addWidget(self.calcBtn)
        self.horizontalLayout_13.addLayout(self.verticalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "START 2.0 Admission Predictor"))
        self.Title.setText(_translate("MainWindow", "START 2.0 Admission Predictor"))
        self.triageLabel.setText(_translate("MainWindow", "Triage Category"))
        self.ambulanceLabel.setText(_translate("MainWindow", "Ambulance"))
        self.ageLabel.setText(_translate("MainWindow", "Age"))
        self.prevAdmitLabel.setText(_translate("MainWindow", "Previous Admission"))
        self.presentingLabel.setText(_translate("MainWindow", "Presenting Problem"))
        self.calcBtn.setText(_translate("MainWindow", "Calculate Admission"))