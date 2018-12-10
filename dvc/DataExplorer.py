"""
DataExplorer

UI to configure a DVC run

Usage:
 DataExplorer.py [ -h ] [ --imagedata=<path> ] [ --spheres=0 ] [ --subvol=10 ]

Options:
 --imagedata=path      input filename
 --spheres=n        whether to show spheres
 --subvol=n         the max size of the subvolume in voxel
 -h       display help
 
"""

import sys
from PyQt5 import QtCore
from PyQt5 import QtGui
#from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QDockWidget
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QStyle
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QTableWidget
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QWidget, QPushButton
from PyQt5.QtWidgets import QFormLayout
import vtk
from ccpi.viewer.QVTKCILViewer import QVTKCILViewer
from ccpi.viewer.QVTKWidget import QVTKWidget
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
from docopt import docopt

# Import linking class to join 2D and 3D viewers
import ccpi.viewer.viewerLinker as vlink
from ccpi.viewer.CILViewer import CILViewer
from ccpi.viewer.CILViewer2D import CILViewer2D

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
        self.setGeometry(50, 50, 1200, 600)

        self.e = ErrorObserver()

        self.toolbar()

        self.frame = QFrame()
        self.vl = QVBoxLayout()
        
        self.vtkWidget = QVTKWidget(
                viewer=CILViewer2D, 
                interactorStyle=vlink.Linked2DInteractorStyle
                )
        self.vtkWidget.viewer.debug = False
        self.iren = self.vtkWidget.getInteractor()
        self.vl.addWidget(self.vtkWidget)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        # init some viewer stuff
        self.pointActorsAdded = self.setupPointCloudPipeline()
                       
        # Add the 3D viewer widget
        self.viewer3DWidget = QVTKWidget(
                viewer=CILViewer, 
                interactorStyle=vlink.Linked3DInteractorStyle
                )
        self.viewer3DWidget.viewer.debug = False
        
        self.Dock3DContents = QWidget()
        self.Dock3DContents.setStyleSheet("background-color: rgb(25,51,101)")
        f_layout3D = QFormLayout(self.Dock3DContents)

        self.Dock3D = QDockWidget(self)
        self.Dock3D.setMinimumWidth(300)
        self.Dock3D.setWindowTitle("3D View")
        self.Dock3D.setFeatures(
                QDockWidget.DockWidgetFloatable | 
                QDockWidget.DockWidgetMovable)
        
        f_layout3D.addWidget(self.viewer3DWidget)
        self.Dock3D.setWidget(self.Dock3DContents)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.Dock3D)


        # Initially link viewers
        self.link2D3D = vlink.ViewerLinker(self.vtkWidget.viewer, 
                                           self.viewer3DWidget.viewer)
        self.link2D3D.setLinkPan(False)
        self.link2D3D.setLinkZoom(False)
        self.link2D3D.setLinkWindowLevel(True)
        self.link2D3D.setLinkSlice(True)
        self.link2D3D.enable()
        
        if self.show3D.isChecked():
            self.Dock3D.show()
        else:
            self.Dock3D.hide()

        # add observer to viewer events
        self.vtkWidget.viewer.style.AddObserver("MouseWheelForwardEvent",
                                                self.updateClippingPlane, 1.9)
        self.vtkWidget.viewer.style.AddObserver("MouseWheelBackwardEvent",
                                                self.updateClippingPlane, 1.9)
        self.vtkWidget.viewer.style.AddObserver('LeftButtonReleaseEvent',
                                                self.OnLeftButtonReleaseEvent, 0.5)
        #self.vtkWidget.viewer.style.AddObserver('KeyPressEvent',
        #                                        self.OnKeyPressEvent, 0.5)
        
        # self.toolbar()

        self.createTableWidget()

        self.statusBar()
        self.setStatusTip('Open file to begin visualisation...')

        self.subvol = 80
        self.displaySpheres = True

        self.pointcloud = []
        
        self.show()

        
    def setupPointCloudPipeline(self):
        self.vtkPointCloud = vtk.vtkPoints()
        self.pointActor = vtk.vtkActor()
        self.vertexActor = vtk.vtkActor()
        self.selectActor = vtk.vtkLODActor()
        self.vertices = vtk.vtkCellArray()
        self.pointActorsAdded = False
        self.pointPolyData = vtk.vtkPolyData()
        self.visPlane = [ vtk.vtkPlane() , vtk.vtkPlane() ] 
        self.planeClipper =  [ vtk.vtkClipPolyData() , vtk.vtkClipPolyData() ]
        
        self.pointMapper = vtk.vtkPolyDataMapper()
        
        self.glyph3D = vtk.vtkGlyph3D()
        self.sphereMapper = vtk.vtkPolyDataMapper()
        self.sphereActor = vtk.vtkActor()
        
        self.selectMapper = vtk.vtkPolyDataMapper()
        return False
        
    def toolbar(self):
        openAction = QAction("Open", self)
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.openFile)

        closeAction = QAction("Close", self)
        closeAction.setShortcut("Ctrl+Q")
        closeAction.triggered.connect(self.close)

        tableAction = QAction("Edit Point Cloud", self)
        tableAction.setShortcut("Ctrl+T")
        tableAction.triggered.connect(self.editPointCloud)

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        fileMenu.addAction(openAction)
        fileMenu.addAction(closeAction)
        fileMenu.addAction(tableAction)
        
        
        self.show3D = QAction("Show 3D View", self)
        self.show3D.setCheckable(True)
        self.show3D.setChecked(False)
        #show3D.setShortcut("Ctrl+T")
        self.show3D.triggered.connect(self.showHide3D)
        viewMenu = mainMenu.addMenu('Visualisation')
        viewMenu.addAction(self.show3D)
        
        # Initialise the toolbar
        self.toolbar = self.addToolBar('Viewer tools')

        # define actions
        openAction = QAction(self.style().standardIcon(
            QStyle.SP_DirOpenIcon), 'Open file', self)
        openAction.triggered.connect(self.openFile)

        saveAction = QAction(self.style().standardIcon(
            QStyle.SP_DialogSaveButton), 'Save current render as PNG', self)
        # saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.saveFile)

        # Add actions to toolbar
        self.toolbar.addAction(openAction)
        self.toolbar.addAction(saveAction)

    def showHide3D(self):
        if self.show3D.isChecked():
            self.Dock3D.show()
        else:
            self.Dock3D.hide()
        
    def openFile(self):
        fn = QFileDialog.getOpenFileNames(self, 'Open File')

        # If the user has pressed cancel, the first element of the tuple will be empty.
        # Quit the method cleanly
        if not fn[0]:
            return
        self.openFileByPath(fn)
        
    def openFileByPath(self, fn):
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
                    self.pointcloud = []
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
                    self.e(
                        '', '', 'When reading multiple files, all files must TIFF formatted.')
                    file = file
                    self.displayFileErrorDialog(file)
                    return

            # Have passed basic test, can attempt to load
            #numpy_image = Converter.tiffStack2numpyEnforceBounds(filenames=filenames)
            #reader = Converter.numpy2vtkImporter(numpy_image)
            # reader.Update()
            reader = vtk.vtkTIFFReader()
            sa = vtk.vtkStringArray()
            #i = 0
            # while (i < 1054):
            for fname in filenames:
                #fname = os.path.join(directory,"8bit-1%04d.tif" % i)
                i = sa.InsertNextValue(fname)

            print("read {} files".format(i))

            reader.SetFileNames(sa)
            reader.Update()

        if self.e.ErrorOccurred():
            self.displayFileErrorDialog(file)

        else:
            self.vtkWidget.viewer.setInput3DData(reader.GetOutput())
            self.viewer3DWidget.viewer.setInput3DData(self.vtkWidget.viewer.img3D)

        self.setStatusTip('Ready')

    def saveFile(self):
        dialog = QFileDialog(self)
        dialog.setAcceptMode(QFileDialog.AcceptSave)

        fn = dialog.getSaveFileName(self, 'Save As', '.', "Images (*.png)")

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
        self.quit()

    def loadPointCloudFromCSV(self, filename):
        print("loadPointCloudFromCSV")
        with open(filename, 'r') as csvfile:
            read = csv.reader(csvfile)

            for row in read:
                # read in only numerical values
                # print (row)
                try:
                    row = list(map(lambda x: float(x), row))
                # print ("reduce " , reduce( lambda x,y: isinstance(x,Number) and \
                #          isinstance(y,Number) , row))
                # if reduce( lambda x,y: isinstance(x,Number) and \
                #          isinstance(y,Number) , row):
                    self.pointcloud.append(row)
                except ValueError as ve:
                    print(ve)

            print(self.pointcloud)

            # load data in the QTableWidget
            self.loadIntoTableWidget(self.pointcloud)
            self.renderPointCloud()

    def addToPointCloud(self):
        if len(self.pointcloud) > 0:
            # 1. Load all of the point coordinates into a vtkPoints.
            # Create the topology of the point (a vertex)
            #for point in self.pointcloud: 
            #    print ("renderPointCloud " , point)
            #    self.addToPointCloud(point)
            # shift the location of the point which has been selected 
            # on the slicing plane as the current view orientation 
            # up 1 slice to be able to see it in front of the slice
            orientation = self.vtkWidget.viewer.GetSliceOrientation()
            
            
                
            vertices = self.vertices
            spacing = self.vtkWidget.viewer.img3D.GetSpacing()
            origin  = self.vtkWidget.viewer.img3D.GetOrigin()
            for count in range(len(self.pointcloud)):
                # expected slice orientation at point selection time
                point = self.pointcloud[count][:]
                select_orientation = [ point[i]-int(point[i]) == 0 for i in range(1,4)].index(True)
                if select_orientation == orientation:
                    beta = 1
                    point[orientation+1] = point[orientation+1] + beta
                    
                p = self.vtkPointCloud.InsertNextPoint(point[1] * spacing[0] - origin[0],
                                                       point[2] * spacing[1] - origin[1],
                                                       point[3] * spacing[2] - origin[2])
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(p)
                
       
#    
    def renderPointCloud(self):
       
        self.addToPointCloud()          
    
        if not self.pointActorsAdded:
            
            # 2. Add the points to a vtkPolyData.
            self.pointPolyData.SetPoints(self.vtkPointCloud)
            self.pointPolyData.SetVerts(self.vertices)
            
            # render the point cloud
            # clipping plane
            #plane = vtk.vtkPlane()
            plane = self.visPlane[0]
            plane2 = self.visPlane[1]
            #point = self.vtkPointCloud
            clipper = self.planeClipper[0]
            clipper2 = self.planeClipper[1]
            #plane.SetOrigin(0, 1., 0)
            #plane.SetNormal(0, 1., 0)

            if not self.displaySpheres:

                #mapper = vtk.vtkPolyDataMapper()
                mapper = self.pointMapper
                mapper.SetInputData(self.pointPolyData)
                actor = self.vertexActor
                actor.SetMapper(mapper)
                actor.GetProperty().SetPointSize(3)
                actor.GetProperty().SetColor(1, .2, .2)
                self.pointActor = actor
                actor.VisibilityOn()
                clipper.SetInputData(self.pointPolyData)
                
               
            else:
                # subvolume
                # arrow
                subv_glyph = self.glyph3D
                subv_glyph.SetScaleFactor(1.)
                # arrow_glyph.SetColorModeToColorByVector()
                sphere_source = vtk.vtkSphereSource()
                spacing = self.vtkWidget.viewer.img3D.GetSpacing()
                radius = self.subvol / max(spacing)
                sphere_source.SetRadius(radius)
                sphere_source.SetThetaResolution(12)
                sphere_source.SetPhiResolution(12)
                sphere_mapper = self.sphereMapper
                sphere_mapper.SetInputConnection(
                    subv_glyph.GetOutputPort())

                subv_glyph.SetInputData(self.pointPolyData)
                subv_glyph.SetSourceConnection(
                    sphere_source.GetOutputPort())

                # Usual actor
                sphere_actor = self.sphereActor
                sphere_actor.SetMapper(sphere_mapper)
                sphere_actor.GetProperty().SetColor(1, 0, 0)
                sphere_actor.GetProperty().SetOpacity(0.2)

                clipper.SetInputConnection(subv_glyph.GetOutputPort())
                
            
            
            clipper.SetClipFunction(plane)
            clipper.InsideOutOn()
            clipper2.SetClipFunction(plane2)
            clipper2.InsideOutOn()
            clipper2.SetInputConnection(clipper.GetOutputPort())

            selectMapper = self.selectMapper
            selectMapper.SetInputConnection(clipper2.GetOutputPort())
            # selectMapper.AddClippingPlane(plane)
            # selectMapper.SetInputData(self.pointPolyData)
            # selectMapper.Update()
            
            selectActor = self.selectActor
            #selectActor = vtk.vtkLODActor()
            selectActor.SetMapper(selectMapper)
            selectActor.GetProperty().SetColor(0, 1, .2)
            selectActor.VisibilityOn()
            #selectActor.SetScale(1.01, 1.01, 1.01)
            selectActor.GetProperty().SetPointSize(5)

            self.vtkWidget.viewer.getRenderer().AddActor(selectActor)
            self.viewer3DWidget.viewer.getRenderer().AddActor(actor)
            
            selectActor3D = vtk.vtkLODActor()
            selectMapper3D = vtk.vtkPolyDataMapper()
            selectMapper3D.SetInputConnection(clipper2.GetOutputPort())
            
            selectActor3D.SetMapper(selectMapper3D)
            selectActor3D.GetProperty().SetColor(0, 1, .2)
            selectActor3D.GetProperty().SetPointSize(5)
            selectActor3D.VisibilityOn()
            
            
            ### plane actors
            # self.setupPlanes()
            
            # self.vtkWidget.viewer.getRenderer().AddActor(actor)
            self.viewer3DWidget.viewer.getRenderer().AddActor(selectActor3D)
            
            
            # self.vtkWidget.viewer.getRenderer().AddActor(sphere_actor)
            print("currently present actors",
                  self.vtkWidget.viewer.getRenderer().GetActors().GetNumberOfItems())
            print("currently present actors",
                  self.viewer3DWidget.viewer.getRenderer().GetActors().GetNumberOfItems())
            self.pointActorsAdded = True
        else:
            print("pointcloud already added")

    def setupPlanes(self):
        self.planesource = [ vtk.vtkPlaneSource(), vtk.vtkPlaneSource() ]
        self.planesource[0].SetCenter(self.visPlane[0].GetOrigin())
        self.planesource[1].SetCenter(self.visPlane[1].GetOrigin())
        self.planesource[0].SetNormal(self.visPlane[0].GetNormal())
        self.planesource[1].SetNormal(self.visPlane[1].GetNormal())
        
        self.planemapper = [ vtk.vtkPolyDataMapper(), vtk.vtkPolyDataMapper() ]
        self.planemapper[0].SetInputData(self.planesource[0].GetOutput())
        self.planemapper[1].SetInputData(self.planesource[1].GetOutput())
        
        self.planeactor = [ vtk.vtkActor(), vtk.vtkActor() ] 
        self.planeactor[0].SetMapper(self.planemapper[0])
        self.planeactor[1].SetMapper(self.planemapper[1])
        
    def updateClippingPlane(self, obj, event):
        # print("caught updateClippingPlane!", event)
        normal = [0, 0, 0]
        origin = [0, 0, 0]
        norm = 1
        orientation = self.vtkWidget.viewer.GetSliceOrientation()
        if orientation == SLICE_ORIENTATION_XY:
            norm = 1
        elif orientation == SLICE_ORIENTATION_XZ:
            norm = -1
        elif orientation == SLICE_ORIENTATION_YZ:
            norm = 1
        beta = 0
        if event == "MouseWheelForwardEvent":
            # this is pretty absurd but it seems the
            # plane cuts too much in Forward...
            beta = +2
        
        spac = self.vtkWidget.viewer.img3D.GetSpacing()
        orig = self.vtkWidget.viewer.img3D.GetOrigin()
        slice_thickness = spac[orientation]
        
        normal[orientation] = norm
        origin [orientation] = (self.vtkWidget.viewer.GetActiveSlice() + beta ) * \
           slice_thickness - orig[orientation]
            
        # print("slice {} beta {} orig {} spac {}".format(self.vtkWidget.viewer.GetActiveSlice(), beta,
        #      orig, spac ))
        # print("origin", origin, orientation)
        # print("<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>")

        self.visPlane[0].SetOrigin(origin[0], origin[1], origin[2])
        self.visPlane[0].SetNormal(normal[0], normal[1], normal[2])
        
        # update the  plane below
        beta += 1
        slice_below = self.vtkWidget.viewer.GetActiveSlice() -1 + beta
        if slice_below < 0:
            slice_below = 0
        origin [orientation] = ( slice_below ) * \
           slice_thickness - orig[orientation]
        self.visPlane[1].SetOrigin(origin[0], origin[1], origin[2])
        
        self.visPlane[1].SetNormal(-normal[0], -normal[1], -normal[2])
        # self.vtkWidget.viewer.sliceActor.VisibilityOff()
        # self.vtkWidget.viewer.sliceActor.VisibilityOn()
        
        
        # for i in range(2):
        #    self.planesource[i].SetCenter(self.visPlane[i].GetOrigin())
        #    self.planesource[i].SetNormal(self.visPlane[i].GetNormal())
        #    self.planemapper[i].Update()
            
        
        # self.viewer3DWidget.viewer.getRenderer().AddActor(self.planeactor[0])
        # self.viewer3DWidget.viewer.getRenderer().AddActor(self.planeactor[1])
        
        # self.vtkWidget.viewer.getRenderer().Render()
        # self.viewer3DWidget.viewer.getRenderer().Render()
        
    def setSubvolSize(self, subvolume):
        self.subvol = subvolume

    def dislayPointCloudAsSpheres(self, should):
        self.displaySpheres = should

    def editPointCloud(self):
        self.tableDock.show()

    def createTableWidget(self):
        self.tableDock = QDockWidget(self)
        self.tableWindow = QMainWindow()
        self.tableWidget = QTableWidget()
        self.tableWindow.setCentralWidget(self.tableWidget)
        self.tableDock.setMinimumWidth(300)
        self.tableDock.setWidget(self.tableWindow)
        self.tableDock.setWindowTitle("Edit Point Cloud")

        sphereAction = QAction("Toggle Sphere visualisation", self)
        sphereAction.setShortcut("Ctrl+S")
        sphereAction.setCheckable(True)
        sphereAction.setChecked(False)
        sphereAction.triggered.connect(
            self.dislayPointCloudAsSpheres, sphereAction.isChecked())

        self.interactiveEdit = QAction("Interactive Edit of Point Cloud", self)
        self.interactiveEdit.setCheckable(True)
        self.interactiveEdit.setChecked(True)

        tableAction = QAction("Update Point Cloud", self)
        tableAction.setShortcut("Ctrl+T")
        tableAction.triggered.connect(self.updatePointCloud)

        mainMenu = self.tableWindow.menuBar()
        fileMenu = mainMenu.addMenu('Edit')
        fileMenu.addAction(self.interactiveEdit)

        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.tableDock)

        self.tableDock.show()
        

    def loadIntoTableWidget(self, data):
        if len(data) <= 0:
            return
        self.tableWidget.setRowCount(len(data))
        self.tableWidget.setColumnCount(len(data[0]))
        for i, v in enumerate(data):
            for j, w in enumerate(v):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(w)))

    def updatePointCloud(self):
        print("should read the table here and save to csv")

    def OnLeftButtonReleaseEvent(self, interactor, event):
        if self.interactiveEdit.isChecked() and self.tableDock.isVisible():

            position = interactor.GetEventPosition()
            print("pick position {}".format(position))
            vox = self.vtkWidget.viewer.style.display2imageCoordinate(position, subvoxel=True)
            print("pick vox {}".format(vox))
            # print("[%d,%d,%d] : %.2g" % vox)
            rows = self.tableWidget.rowCount()
            cols = self.tableWidget.columnCount()

            self.tableWidget.setRowCount(rows + 1)
            if cols != 4:
                self.tableWidget.setColumnCount(4)
            for col in range(3):
                self.tableWidget.setItem(rows, col+1,
                                         QTableWidgetItem(str(vox[col])))
            rows = self.tableWidget.rowCount()
            print("rows", rows)
            if rows == 1:
                el = 1
            else:
                print("row {0} el {1} ".format(
                    rows, self.tableWidget.item(rows-2, 0).text()))
                el = int(self.tableWidget.item(rows-2, 0).text())
            self.tableWidget.setItem(rows-1, 0, QTableWidgetItem(str(el+1)))

            # self.pointcloud.append([el+1, vox[0], vox[1], vox[2]])

            if not self.pointActorsAdded:
                self.pointcloud.append([el+1, vox[0], vox[1], vox[2]])
                self.renderPointCloud()
            else:
                self.appendPointToCloud([el+1, vox[0], vox[1], vox[2]])
                #self.renderPointCloud()
            self.vtkWidget.viewer.getRenderer().Render()

    def appendPointToCloud(self, point):
        self.pointcloud.append(point)
        self.pointActorsAdded = self.setupPointCloudPipeline()
        self.renderPointCloud()
        
    
        

def main():
    err = vtk.vtkFileOutputWindow()
    err.SetFileName("viewer.log")
    vtk.vtkOutputWindow.SetInstance(err)
    
    __version__ = '0.1.0'
    print ("Starting ... ")
    args = docopt(__doc__, version=__version__)
    print ("Parsing args")
    print ("Passed args " , args)
    
    App = QApplication(sys.argv)
    gui = Window()
    
    show_spheres = False
    if not args['--spheres'] is None:
        show_spheres = True if args["--spheres"] == 1 else False
        
    subvol_size = 80
    if not args['--subvol'] is None:
        subvol_size = int(args["--subvol"])
    
    if not args['--imagedata'] is None:
        fname = os.path.abspath(args["--imagedata"])
        gui.openFileByPath(( (fname , ),))
    
    gui.setSubvolSize(subvol_size)
    gui.dislayPointCloudAsSpheres(show_spheres)
    
        
    sys.exit(App.exec())
    
     
    

if __name__ == "__main__":
    main()
