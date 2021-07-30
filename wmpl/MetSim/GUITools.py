
""" Matplotlib widget for PyQt5. """

from PyQt5.QtWidgets import*
from PyQt5 import QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure




# Top figure sizes
TOP_FIGSIZES = (3, 5)

    
class MagnitudeMplWidget(QWidget):
    
    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        
        # Create the figure
        self.canvas = FigureCanvas(Figure(facecolor='#efebe7', figsize=(7, 5)))
        self.canvas.axes = self.canvas.figure.add_subplot(111)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar2QT(self.canvas, self)

        # Create a layout and add the navigation and plot
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(toolbar)
        vertical_layout.addWidget(self.canvas)

        self.setLayout(vertical_layout)



class LagMplWidget(QWidget):
    
    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        
        # Create the figure
        self.canvas = FigureCanvas(Figure(facecolor='#efebe7', figsize=(7, 5)))
        self.canvas.axes = self.canvas.figure.add_subplot(111)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar2QT(self.canvas, self)

        # Create a layout and add the navigation and plot
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(toolbar)
        vertical_layout.addWidget(self.canvas)

        self.setLayout(vertical_layout)


class VelocityMplWidget(QWidget):
    
    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        
        # Create the figure
        self.canvas = FigureCanvas(Figure(facecolor='#efebe7', figsize=(7, 5)))
        self.canvas.axes = self.canvas.figure.add_subplot(111)

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar2QT(self.canvas, self)

        # Create a layout and add the navigation and plot
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(toolbar)
        vertical_layout.addWidget(self.canvas)

        self.setLayout(vertical_layout)



class WakeMplWidget(QWidget):
    
    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        
        # Create the figure
        self.canvas = FigureCanvas(Figure(facecolor='#efebe7', figsize=(8, 5)))
        self.canvas.axes = self.canvas.figure.add_subplot(111)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        
        self.setLayout(vertical_layout)



class MatplotlibPopupWindow(QMainWindow):
    def __init__(self):
        super(MatplotlibPopupWindow, self).__init__()
        
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        
        
        
        # Init the matplotlib canvas
        self.canvas = FigureCanvas(Figure(facecolor='#efebe7'))
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        


        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar2QT(self.canvas, self)

        # Compose elements into layout
        layout = QVBoxLayout(self.main_widget)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)