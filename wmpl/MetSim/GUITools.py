
""" Matplotlib widget for PyQt5. """

from PyQt5.QtWidgets import*

from matplotlib.backends.backend_qt5agg import FigureCanvas

from matplotlib.figure import Figure


# Top figure sizes
TOP_FIGSIZES = (3, 5)

    
class MagnitudeMplWidget(QWidget):
    
    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        
        self.canvas = FigureCanvas(Figure(facecolor='#efebe7', figsize=TOP_FIGSIZES))
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)



class LagMplWidget(QWidget):
    
    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        
        self.canvas = FigureCanvas(Figure(facecolor='#efebe7', figsize=TOP_FIGSIZES))
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)


class VelocityMplWidget(QWidget):
    
    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        
        self.canvas = FigureCanvas(Figure(facecolor='#efebe7', figsize=TOP_FIGSIZES))
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)



class WakeMplWidget(QWidget):
    
    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        
        self.canvas = FigureCanvas(Figure(facecolor='#efebe7'))
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)