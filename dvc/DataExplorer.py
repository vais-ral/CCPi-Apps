import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
import vtk
from ccpi.viewer.QVTKCILViewer import QVTKCILViewer
from ccpi.viewer.CILViewer2D import Converter
from ccpi.viewer.CILViewer2D import SLICE_ORIENTATION_XY
from ccpi.viewer.CILViewer2D import SLICE_ORIENTATION_XZ
from ccpi.viewer.CILViewer2D import SLICE_ORIENTATION_YZ
from natsort import natsorted
import imghdr
import os
import csv
from functools import reduce
from numbers import Number


class ErrorObserver:

   def __init__(self):
       self.__ErrorOccurred = False
       self.__ErrorMessage = None
       self.CallDataType = 'string0'

   def __call__(self, obj, event, message):
       self.__ErrorOccurred = True
       self.__ErrorMessage = message

   def ErrorOccurred(self):
       occ = self.__ErrorOccurred
       self.__ErrorOccurred = False
       return occ

   def ErrorMessage(self):
       return self.__ErrorMessage


def sentenceCase(string):
    if string:
        first_word = string.split()[0]
        world_len = len(first_word)

        return first_word.capitalize() + string[world_len:]

    else:
        return ''



class Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('CIL Viewer')
        self.setGeometry(50, 50, 600, 600)

        self.e = ErrorObserver()

        openAction = QAction("Open", self)
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.openFile)

        closeAction = QAction("Close", self)
        closeAction.setShortcut("Ctrl+Q")
        closeAction.triggered.connect(self.close)

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        fileMenu.addAction(openAction)
        fileMenu.addAction(closeAction)

        self.frame = QFrame()
        self.vl = QVBoxLayout()
        self.vtkWidget = QVTKCILViewer(self.frame)
        self.iren = self.vtkWidget.getInteractor()
        self.vl.addWidget(self.vtkWidget)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        
        # init some viewer stuff
        self.vtkPointCloud = vtk.vtkPoints()
        self.pointActor = vtk.vtkActor()
        self.selectActor = vtk.vtkLODActor()
        self.pointPolyData = vtk.vtkPolyData()
        self.visPlane = vtk.vtkPlane()
        self.planeClipper = vtk.vtkClipPolyData()
        self.vtkWidget.viewer.style.AddObserver("MouseWheelForwardEvent", 
                                                self.updateClippingPlane, 0.9)
        self.vtkWidget.viewer.style.AddObserver("MouseWheelBackwardEvent", 
                                                self.updateClippingPlane, 0.9)
        self.toolbar()

        self.statusBar()
        self.setStatusTip('Open file to begin visualisation...')

        self.show()

    def toolbar(self):
        # Initialise the toolbar
        self.toolbar = self.addToolBar('Viewer tools')

        # define actions
        openAction = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), 'Open file', self)
        openAction.triggered.connect(self.openFile)

        saveAction = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), 'Save current render as PNG', self)
        # saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.saveFile)

        # Add actions to toolbar
        self.toolbar.addAction(openAction)
        self.toolbar.addAction(saveAction)
        
        


    def openFile(self):
        self.pointcloud = []
        fn = QFileDialog.getOpenFileNames(self, 'Open File')

        # If the user has pressed cancel, the first element of the tuple will be empty.
        # Quit the method cleanly
        if not fn[0]:
            return

        # Single file selection
        if len(fn[0]) == 1:
            file = fn[0][0]
            
            if imghdr.what(file) == None:
                if file.split(".")[1] == 'mha' or\
                    file.split(".")[1] == 'mhd':
                    reader = vtk.vtkMetaImageReader()
                    reader.AddObserver("ErrorEvent", self.e)
                    reader.SetFileName(file)
                    reader.Update()
                elif file.split(".")[1] == 'csv':
                    self.loadPointCloudFromCSV(file)
                    return
            else:
                return
        # Multiple TIFF files selected
        else:
            # Make sure that the files are sorted 0 - end
            filenames = natsorted(fn[0])

            # Basic test for tiff images
            for file in filenames:
                ftype = imghdr.what(file)
                if ftype != 'tiff':
                    # A non-TIFF file has been loaded, present error message and exit method
                    self.e('','','When reading multiple files, all files must TIFF formatted.')
                    file = file
                    self.displayFileErrorDialog(file)
                    return

            # Have passed basic test, can attempt to load
            #numpy_image = Converter.tiffStack2numpyEnforceBounds(filenames=filenames)
            #reader = Converter.numpy2vtkImporter(numpy_image)
            #reader.Update()
            reader = vtk.vtkTIFFReader()
            sa = vtk.vtkStringArray()
            #i = 0
            #while (i < 1054):
            for fname in filenames:
                #fname = os.path.join(directory,"8bit-1%04d.tif" % i)
                i = sa.InsertNextValue(fname)
                
            print ("read {} files".format( i ))
            
            reader.SetFileNames(sa)
            reader.Update()

        if self.e.ErrorOccurred():
            self.displayFileErrorDialog(file)

        else:
            self.vtkWidget.viewer.setInput3DData(reader.GetOutput())

        self.setStatusTip('Ready')


    def saveFile(self):
        dialog = QFileDialog(self)
        dialog.setAcceptMode(QFileDialog.AcceptSave)

        fn = dialog.getSaveFileName(self,'Save As','.',"Images (*.png)")

        # Only save if the user has selected a name
        if fn[0]:
            self.vtkWidget.viewer.saveRender(fn[0])

    def displayFileErrorDialog(self, file):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("READ ERROR")
        msg.setText("Error reading file: ({filename})".format(filename=file))
        msg.setDetailedText(self.e.ErrorMessage())
        msg.exec_()


    def close(self):
        qApp.quit()
        
    def loadPointCloudFromCSV(self, filename):
        print ("loadPointCloudFromCSV")
        with open(filename, 'r') as csvfile:
            read = csv.reader(csvfile)
            i = 0
            for row in read:
                #read in only numerical values
                print (row)
                try:
                    row = list(map(lambda x: float(x),row))
                #print ("reduce " , reduce( lambda x,y: isinstance(x,Number) and \
                #          isinstance(y,Number) , row))
                #if reduce( lambda x,y: isinstance(x,Number) and \
                #          isinstance(y,Number) , row):
                    self.pointcloud.append(row)
                except ValueError as ve:
                    print (ve)
                    
            print (self.pointcloud)
            # render the point cloud
            if len(self.pointcloud) > 0:
                #1. Load all of the point coordinates into a vtkPoints.
                # Create the topology of the point (a vertex)
                vertices = vtk.vtkCellArray()
                
                for count in range(len(self.pointcloud)):
                    p = self.vtkPointCloud.InsertNextPoint(self.pointcloud[count][1],
                                                           self.pointcloud[count][2], 
                                                           self.pointcloud[count][3])
                    vertices.InsertNextCell(1)
                    vertices.InsertCellPoint(p)
                    
    
                #2. Add the points to a vtkPolyData.
                self.pointPolyData.SetPoints( self.vtkPointCloud ) 
                self.pointPolyData.SetVerts( vertices ) 
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(self.pointPolyData)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetPointSize(3)
                actor.GetProperty().SetColor(0,1,1)
                self.pointActor = actor
                actor.VisibilityOn()
                
                #plane = vtk.vtkPlane()
                plane = self.visPlane
                point = self.vtkPointCloud
                clipper = self.planeClipper
                plane.SetOrigin(0,1.,0)
                plane.SetNormal(0,1,0)
                
                
                #clipper.SetInputConnection(apd.GetOutputPort())
                clipper.SetInputData(self.pointPolyData)
                clipper.SetClipFunction(plane)
                clipper.InsideOutOn()
                
                selectMapper = vtk.vtkPolyDataMapper()
                selectMapper.SetInputConnection(clipper.GetOutputPort())
                
                selectActor = self.selectActor
                #selectActor = vtk.vtkLODActor()
                selectActor.SetMapper(selectMapper)
                selectActor.GetProperty().SetColor(0, 1, 0)
                selectActor.VisibilityOn()
                selectActor.SetScale(1.01, 1.01, 1.01)
                selectActor.GetProperty().SetPointSize(20)

                
                self.vtkWidget.viewer.getRenderer().AddActor(actor)
                self.vtkWidget.viewer.getRenderer().AddActor(selectActor)
                print ("currently present actors" , 
                       self.vtkWidget.viewer.getRenderer().GetActors().GetNumberOfItems())
                
                
    def updateClippingPlane(self, obj, event):
        print ("caught updateClippingPlane!", event)
        normal = [0,0,0]
        origin = [0,0,0]
        orientation = self.vtkWidget.viewer.GetSliceOrientation()
        norm = 1
        if orientation == SLICE_ORIENTATION_XY:
            norm = 1
        elif orientation == SLICE_ORIENTATION_XZ:
            norm = -1
        elif orientation == SLICE_ORIENTATION_YZ:
            norm = 1
        normal[orientation] = norm
        origin[orientation] = self.vtkWidget.viewer.GetActiveSlice() + 0.1 * norm
        print ("normal" , normal)
        print ("origin" , origin)
        plane = self.visPlane
        plane.SetOrigin(origin[0],origin[1],origin[2])
        plane.SetNormal(normal[0],normal[1],normal[2])
        
def main():
    err = vtk.vtkFileOutputWindow()
    err.SetFileName("viewer.log")
    vtk.vtkOutputWindow.SetInstance(err)
    
    App = QApplication(sys.argv)
    gui = Window()
    sys.exit(App.exec())

if __name__=="__main__":
    main()