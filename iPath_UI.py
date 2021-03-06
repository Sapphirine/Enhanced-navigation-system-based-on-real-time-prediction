# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(QtWidgets.QMainWindow):
    def setupUi(self, Form):
        Form.setObjectName("iPath")
        Form.resize(721, 553)
        self.gridLayout_8 = QtWidgets.QGridLayout(Form)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout_3.addWidget(self.graphicsView)
        self.graphicsView_2 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.verticalLayout_3.addWidget(self.graphicsView_2)
        self.gridLayout_8.addLayout(self.verticalLayout_3, 0, 0, 2, 1)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.speed = QtWidgets.QComboBox(Form)
        self.speed.setObjectName("speed")
        self.speed.addItem("")
        self.speed.addItem("")
        self.speed.addItem("")
        self.speed.addItem("")
        self.speed.addItem("")
        self.gridLayout_6.addWidget(self.speed, 0, 1, 1, 1)
        self.label_destination = QtWidgets.QLabel(Form)
        self.label_destination.setObjectName("label_destination")
        self.gridLayout_6.addWidget(self.label_destination, 2, 0, 1, 1)
        self.label_speed = QtWidgets.QLabel(Form)
        self.label_speed.setObjectName("label_speed")
        self.gridLayout_6.addWidget(self.label_speed, 0, 0, 1, 1)
        self.destination_y = QtWidgets.QComboBox(Form)
        self.destination_y.setObjectName("destination_y")
        self.destination_y.addItem("")
        self.destination_y.addItem("")
        self.destination_y.addItem("")
        self.destination_y.addItem("")
        self.destination_y.addItem("")
        self.destination_y.addItem("")
        self.destination_y.addItem("")
        self.destination_y.addItem("")
        self.destination_y.addItem("")
        self.destination_y.addItem("")
        self.gridLayout_6.addWidget(self.destination_y, 2, 2, 1, 1)
        self.destination_x = QtWidgets.QComboBox(Form)
        self.destination_x.setObjectName("destination_x")
        self.destination_x.addItem("")
        self.destination_x.addItem("")
        self.destination_x.addItem("")
        self.destination_x.addItem("")
        self.destination_x.addItem("")
        self.destination_x.addItem("")
        self.destination_x.addItem("")
        self.destination_x.addItem("")
        self.destination_x.addItem("")
        self.destination_x.addItem("")
        self.gridLayout_6.addWidget(self.destination_x, 2, 1, 1, 1)
        self.departure_y = QtWidgets.QComboBox(Form)
        self.departure_y.setObjectName("departure_y")
        self.departure_y.addItem("")
        self.departure_y.addItem("")
        self.departure_y.addItem("")
        self.departure_y.addItem("")
        self.departure_y.addItem("")
        self.departure_y.addItem("")
        self.departure_y.addItem("")
        self.departure_y.addItem("")
        self.departure_y.addItem("")
        self.departure_y.addItem("")
        self.gridLayout_6.addWidget(self.departure_y, 1, 2, 1, 1)
        self.departure_x = QtWidgets.QComboBox(Form)
        self.departure_x.setObjectName("departure_x")
        self.departure_x.addItem("")
        self.departure_x.addItem("")
        self.departure_x.addItem("")
        self.departure_x.addItem("")
        self.departure_x.addItem("")
        self.departure_x.addItem("")
        self.departure_x.addItem("")
        self.departure_x.addItem("")
        self.departure_x.addItem("")
        self.departure_x.addItem("")
        self.gridLayout_6.addWidget(self.departure_x, 1, 1, 1, 1)
        self.label_departure = QtWidgets.QLabel(Form)
        self.label_departure.setObjectName("label_departure")
        self.gridLayout_6.addWidget(self.label_departure, 1, 0, 1, 1)
        self.verticalLayout_5.addLayout(self.gridLayout_6)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.start = QtWidgets.QPushButton(Form)
        self.start.setObjectName("start")
        self.gridLayout_2.addWidget(self.start, 2, 0, 1, 1)
        self.stop = QtWidgets.QPushButton(Form)
        self.stop.setObjectName("stop")
        self.gridLayout_2.addWidget(self.stop, 2, 1, 1, 1)
        self.a_star = QtWidgets.QPushButton(Form)
        self.a_star.setObjectName("a_star")
        self.gridLayout_2.addWidget(self.a_star, 0, 0, 1, 1)
        self.dijkstra = QtWidgets.QPushButton(Form)
        self.dijkstra.setObjectName("dijkstra")
        self.gridLayout_2.addWidget(self.dijkstra, 0, 1, 1, 1)
        self.reset = QtWidgets.QPushButton(Form)
        self.reset.setObjectName("reset")
        self.gridLayout_2.addWidget(self.reset, 3, 0, 1, 1)
        self.back = QtWidgets.QPushButton(Form)
        self.back.setObjectName("back")
        self.gridLayout_2.addWidget(self.back, 3, 1, 1, 1)
        self.verticalLayout_5.addLayout(self.gridLayout_2)
        self.gridLayout_8.addLayout(self.verticalLayout_5, 0, 1, 1, 1)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.gridLayout_8.addLayout(self.gridLayout_7, 1, 1, 1, 1)

        self.retranslateUi(Form)
        self.back.clicked.connect(Form.close)
        self.speed.currentIndexChanged['int'].connect(self.graphicsView.update)
        self.departure_x.currentIndexChanged['int'].connect(self.graphicsView.update)
        self.departure_y.currentIndexChanged['int'].connect(self.graphicsView.update)
        self.destination_x.currentIndexChanged['int'].connect(self.graphicsView.update)
        self.destination_y.currentIndexChanged['int'].connect(self.graphicsView.update)
        self.a_star.clicked.connect(self.graphicsView.update)
        self.dijkstra.clicked.connect(self.graphicsView.update)
        self.start.clicked.connect(self.graphicsView.update)
        self.stop.clicked.connect(self.graphicsView.update)
        self.reset.clicked.connect(self.graphicsView.update)
        QtCore.QMetaObject.connectSlotsByName(Form)

        Form.show()

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.speed.setItemText(0, _translate("Form", "1"))
        self.speed.setItemText(1, _translate("Form", "2"))
        self.speed.setItemText(2, _translate("Form", "3"))
        self.speed.setItemText(3, _translate("Form", "4"))
        self.speed.setItemText(4, _translate("Form", "5"))
        self.label_destination.setText(_translate("Form", "destination"))
        self.label_speed.setText(_translate("Form", "speed"))
        self.destination_y.setItemText(0, _translate("Form", "1"))
        self.destination_y.setItemText(1, _translate("Form", "2"))
        self.destination_y.setItemText(2, _translate("Form", "3"))
        self.destination_y.setItemText(3, _translate("Form", "4"))
        self.destination_y.setItemText(4, _translate("Form", "5"))
        self.destination_y.setItemText(5, _translate("Form", "6"))
        self.destination_y.setItemText(6, _translate("Form", "7"))
        self.destination_y.setItemText(7, _translate("Form", "8"))
        self.destination_y.setItemText(8, _translate("Form", "9"))
        self.destination_y.setItemText(9, _translate("Form", "10"))
        self.destination_x.setItemText(0, _translate("Form", "1"))
        self.destination_x.setItemText(1, _translate("Form", "2"))
        self.destination_x.setItemText(2, _translate("Form", "3"))
        self.destination_x.setItemText(3, _translate("Form", "4"))
        self.destination_x.setItemText(4, _translate("Form", "5"))
        self.destination_x.setItemText(5, _translate("Form", "6"))
        self.destination_x.setItemText(6, _translate("Form", "7"))
        self.destination_x.setItemText(7, _translate("Form", "8"))
        self.destination_x.setItemText(8, _translate("Form", "9"))
        self.destination_x.setItemText(9, _translate("Form", "10"))
        self.departure_y.setItemText(0, _translate("Form", "1"))
        self.departure_y.setItemText(1, _translate("Form", "2"))
        self.departure_y.setItemText(2, _translate("Form", "3"))
        self.departure_y.setItemText(3, _translate("Form", "4"))
        self.departure_y.setItemText(4, _translate("Form", "5"))
        self.departure_y.setItemText(5, _translate("Form", "6"))
        self.departure_y.setItemText(6, _translate("Form", "7"))
        self.departure_y.setItemText(7, _translate("Form", "8"))
        self.departure_y.setItemText(8, _translate("Form", "9"))
        self.departure_y.setItemText(9, _translate("Form", "10"))
        self.departure_x.setItemText(0, _translate("Form", "1"))
        self.departure_x.setItemText(1, _translate("Form", "2"))
        self.departure_x.setItemText(2, _translate("Form", "3"))
        self.departure_x.setItemText(3, _translate("Form", "4"))
        self.departure_x.setItemText(4, _translate("Form", "5"))
        self.departure_x.setItemText(5, _translate("Form", "6"))
        self.departure_x.setItemText(6, _translate("Form", "7"))
        self.departure_x.setItemText(7, _translate("Form", "8"))
        self.departure_x.setItemText(8, _translate("Form", "9"))
        self.departure_x.setItemText(9, _translate("Form", "10"))
        self.label_departure.setText(_translate("Form", "departure"))
        self.start.setText(_translate("Form", "start"))
        self.stop.setText(_translate("Form", "stop"))
        self.a_star.setText(_translate("Form", "A*"))
        self.dijkstra.setText(_translate("Form", "Dijkstra"))
        self.reset.setText(_translate("Form", "reset"))
        self.back.setText(_translate("Form", "go back to iPath"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget(None)
    Ui_Form().setupUi(widget)
    sys.exit(app.exec_())
    pass
