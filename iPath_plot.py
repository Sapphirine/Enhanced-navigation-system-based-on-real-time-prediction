# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFrame, QDesktopWidget, QApplication
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor
import sys


class iPath(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        '''initiates application UI'''

        self.roads = control(self)
        self.setCentralWidget(self.roads)
        self.roads.start()

        self.resize(500, 500)
        self.setWindowTitle('iPath')
        self.show()


class control(QFrame):
    BoardWidth = 25
    BoardHeight = 25
    Speed = 300

    def __init__(self, parent):
        super().__init__(parent)

        self.initControl()

    def initControl(self):

        self.timer = QBasicTimer()
        self.curX = 0
        self.curY = 0
        self.board = [0 for i in range(control.BoardHeight * control.BoardWidth)]
        self.clearBoard()

    def squareWidth(self):
        return self.contentsRect().width() // control.BoardWidth

    def squareHeight(self):
        return self.contentsRect().height() // control.BoardHeight

    def shapeAt(self, x, y):

        return self.board[(y * control.BoardWidth) + x]

    def setShapeAt(self, x, y, shape):

        self.board[(y * control.BoardWidth) + x] = shape

    def start(self):
        self.clearBoard()
        self.newGraph()
        self.timer.start(control.Speed, self)

    def timerEvent(self, event):

        if event.timerId() == self.timer.timerId():
            pass
            # self.newGraph()
            # self.update()
        else:
            super(control, self).timerEvent(event)

    def clearBoard(self):

        for i in range(control.BoardHeight * control.BoardWidth):
            self.board[i] = Road.NoShape

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.contentsRect()
        boardTop = rect.bottom() - control.BoardHeight * self.squareHeight()

        for i in range(control.BoardHeight):
            for j in range(control.BoardWidth):
                shape = self.shapeAt(j, control.BoardHeight - i - 1)
                if shape != Road.NoShape:
                    if shape == Road.dot:
                        self.drawRoad(painter,
                            rect.left() + j * self.squareWidth(),
                            boardTop + i * self.squareHeight(), shape, 0)
                    else:
                        self.drawRoad(painter,
                            rect.left() + j * self.squareWidth(),
                            boardTop + i * self.squareHeight(), shape, 3)

    def drawRoad(self, painter, x, y, shape, colour):

        colorTable = [0xFFFFFF, 0xCC6666, 0x66CC66, 0xCCCC66]  # white red green yellow

        color = QColor(colorTable[colour])
        painter.fillRect(x + 1, y + 1, self.squareWidth() - 2,
                         self.squareHeight() - 2, color)

        painter.setPen(color.lighter())
        painter.drawLine(x, y + self.squareHeight() - 1, x, y)
        painter.drawLine(x, y, x + self.squareWidth() - 1, y)

        painter.setPen(color.darker())
        painter.drawLine(x + 1, y + self.squareHeight() - 1,
                         x + self.squareWidth() - 1, y + self.squareHeight() - 1)
        painter.drawLine(x + self.squareWidth() - 1,
                         y + self.squareHeight() - 1, x + self.squareWidth() - 1, y + 1)

    def newGraph(self):
        cnt1 = 0
        cnt2 = 0
        d_n = {}
        d_e = {}
        for i in range(control.BoardHeight * control.BoardWidth):
            temp = (Map.NODE[cnt1].split('N'))
            if (i % control.BoardWidth % 4 == 0) & (int(i / control.BoardHeight) % 4 == 0):
                self.board[i] = Road.dot
                d_n[Map.NODE[cnt1]] = i
                cnt1 += 1
        for i in range(control.BoardHeight * control.BoardWidth):
            # self.board[i] = Road.VerticalShape
            if i % control.BoardWidth % 4 == 0:
                if (int(i / control.BoardHeight)+2) % 4 == 0:
                    self.board[i] = Road.VerticalShape
                    self.board[i+25] = Road.VerticalShape
                    self.board[i-25] = Road.VerticalShape
                    d_e[Map.EDGE[cnt2]] = i
                    cnt2 += 1
            if (int(i / control.BoardHeight) % 4 == 0) & (i % 2 == 0):
                if (i+2) % control.BoardWidth % 4 == 0:
                    self.board[i] = Road.HorizontalShape
                    self.board[i+1] = Road.HorizontalShape
                    self.board[i-1] = Road.HorizontalShape
                    d_e[Map.EDGE[cnt2]] = i
                    cnt2 += 1


class Road(object):
    NoShape = 0
    VerticalShape = 1
    HorizontalShape = 2
    dot = 3


class Shape(object):

    coordsTable = (
        ((0, 0),     (0, 0),     (0, 0)),
        ((0, -1),    (0, 0),     (0, 1)),
        ((-1, 0),    (0, 0),     (1, 0)),
         (0, 0),     (0, 0),     (0, 0)
    )

    def __init__(self):

        self.coords = [[0,0] for i in range(3)]
        self.pieceShape = Road.NoShape
        self.setShape(Road.NoShape)

    def shape(self):

        return self.pieceShape

    def setShape(self, shape):

        table = Shape.coordsTable[shape]

        for i in range(3):
            for j in range(2):
                self.coords[i][j] = table[i][j]

        self.pieceShape = shape

    def x(self, index):

        return self.coords[index][0]

    def y(self, index):

        return self.coords[index][1]


class Map(object):
    NODE = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7',
            'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14',
            'N15', 'N16', 'N17', 'N18', 'N19', 'N20', 'N21',
            'N22', 'N23', 'N24', 'N25', 'N26', 'N27', 'N28',
            'N29', 'N30', 'N31', 'N32', 'N33', 'N34', 'N35',
            'N36', 'N37', 'N38', 'N39', 'N40', 'N41', 'N42',
            'N43', 'N44', 'N45', 'N46', 'N47', 'N48', 'N49',]
    EDGE = [('N1', 'N2'), ('N2', 'N3'), ('N3', 'N4'), ('N4', 'N5'), ('N5', 'N6'), ('N6', 'N7'),
            ('N1', 'N8'), ('N2', 'N9'), ('N3', 'N10'), ('N4', 'N11'), ('N5', 'N12'), ('N6', 'N13'),('N7', 'N14'),
            ('N8', 'N9'), ('N9', 'N10'), ('N10', 'N11'), ('N11', 'N12'), ('N12', 'N13'), ('N13', 'N14'),
            ('N8', 'N15'), ('N9', 'N16'), ('N10', 'N17'), ('N11', 'N18'), ('N12', 'N19'), ('N13','N20'),('N14','N21'),
            ('N15', 'N16'), ('N16', 'N17'), ('N17', 'N18'), ('N18', 'N19'), ('N19', 'N20'), ('N20', 'N21'),
            ('N15', 'N22'), ('N16', 'N23'),('N17', 'N24'),('N18', 'N25'),('N19', 'N26'),('N20', 'N27'),('N21', 'N28'),
            ('N22', 'N23'), ('N23', 'N24'), ('N124', 'N25'), ('N125', 'N26'), ('N26', 'N27'), ('N27', 'N28'),
            ('N22', 'N29'),('N23', 'N30'),('N24', 'N31'), ('N25', 'N32'), ('N26', 'N33'), ('N27', 'N34'),('N28', 'N35'),
            ('N29', 'N30'), ('N30', 'N31'), ('N31', 'N32'), ('N32', 'N33'), ('N33', 'N34'), ('N34', 'N35'),
            ('N29', 'N36'),('N30', 'N37'),('N31', 'N38'),('N32', 'N39'),('N33', 'N40'), ('N34', 'N41'),('N35', 'N42'),
            ('N36', 'N37'), ('N37', 'N38'), ('N38', 'N39'), ('N39', 'N40'), ('N40', 'N41'), ('N41', 'N42'),
            ('N36', 'N43'),('N37', 'N44'),('N38', 'N45'),('N39', 'N46'),('N40', 'N47'), ('N41', 'N48'),('N42', 'N49'),
            ('N43', 'N44'), ('N44','45'),('N45', 'N46'), ('N46', 'N47'), ('N47', 'N48'), ('N48', 'N49')]


if __name__ == "__main__":
    app = QApplication([])
    ipath = iPath()
    sys.exit(app.exec_())
