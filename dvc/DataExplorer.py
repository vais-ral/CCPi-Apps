import sys
from PyQt5 import QtCore
#from PyQt5 import QtGui
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

        tableAction = QAction("Edit Point Cloud", self)
        tableAction.setShortcut("Ctrl+T")
        tableAction.triggered.connect(self.editPointCloud)

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        fileMenu.addAction(openAction)
        fileMenu.addAction(closeAction)
        fileMenu.addAction(tableAction)

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
        self.vertices = vtk.vtkCellArray()
        self.pointActorsAdded = False
        self.pointPolyData = vtk.vtkPolyData()
        self.visPlane = vtk.vtkPlane()
        self.planeClipper = vtk.vtkClipPolyData()
        # add observer to viewer events
        self.vtkWidget.viewer.style.AddObserver("MouseWheelForwardEvent",
                                                self.updateClippingPlane, 0.9)
        self.vtkWidget.viewer.style.AddObserver("MouseWheelBackwardEvent",
                                                self.updateClippingPlane, 0.9)
        self.vtkWidget.viewer.style.AddObserver('LeftButtonReleaseEvent',
                                                self.OnLeftButtonReleaseEvent, 0.5)
        self.toolbar()

        self.createTableWidget()

        self.statusBar()
        self.setStatusTip('Open file to begin visualisation...')

        self.subvol = 80
        self.displaySpheres = True

        self.show()

    def toolbar(self):
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

    def renderPointCloud(self):
        if not self.pointActorsAdded:
            # render the point cloud
            if len(self.pointcloud) > 0:
                # 1. Load all of the point coordinates into a vtkPoints.
                # Create the topology of the point (a vertex)

                vertices = self.vertices

                for count in range(len(self.pointcloud)):
                    p = self.vtkPointCloud.InsertNextPoint(self.pointcloud[count][1],
                                                           self.pointcloud[count][2],
                                                           self.pointcloud[count][3])
                    vertices.InsertNextCell(1)
                    vertices.InsertCellPoint(p)

                # 2. Add the points to a vtkPolyData.
                self.pointPolyData.SetPoints(self.vtkPointCloud)
                self.pointPolyData.SetVerts(vertices)

                # clipping plane
                #plane = vtk.vtkPlane()
                plane = self.visPlane
                #point = self.vtkPointCloud
                clipper = self.planeClipper
                plane.SetOrigin(0, 1., 0)
                plane.SetNormal(0, 1, 0)

                if not self.displaySpheres:

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(self.pointPolyData)
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetPointSize(3)
                    actor.GetProperty().SetColor(0, 1, 1)
                    self.pointActor = actor
                    actor.VisibilityOff()
                    clipper.SetInputData(self.pointPolyData)

                else:
                    # subvolume
                    # arrow
                    subv_glyph = vtk.vtkGlyph3D()
                    subv_glyph.SetScaleFactor(1.)
                    # arrow_glyph.SetColorModeToColorByVector()
                    sphere_source = vtk.vtkSphereSource()
                    sphere_source.SetRadius(self.subvol)
                    sphere_source.SetThetaResolution(12)
                    sphere_source.SetPhiResolution(12)
                    sphere_mapper = vtk.vtkPolyDataMapper()
                    sphere_mapper.SetInputConnection(
                        subv_glyph.GetOutputPort())

                    subv_glyph.SetInputData(self.pointPolyData)
                    subv_glyph.SetSourceConnection(
                        sphere_source.GetOutputPort())

                    # Usual actor
                    sphere_actor = vtk.vtkActor()
                    sphere_actor.SetMapper(sphere_mapper)
                    sphere_actor.GetProperty().SetColor(1, 0, 0)
                    sphere_actor.GetProperty().SetOpacity(0.2)

                    clipper.SetInputConnection(subv_glyph.GetOutputPort())
                #
                clipper.SetClipFunction(plane)
                clipper.InsideOutOn()

                selectMapper = vtk.vtkPolyDataMapper()
                selectMapper.SetInputConnection(clipper.GetOutputPort())

                selectActor = self.selectActor
                #selectActor = vtk.vtkLODActor()
                selectActor.SetMapper(selectMapper)
                selectActor.GetProperty().SetColor(0, 1, 0)
                selectActor.VisibilityOn()
                #selectActor.SetScale(1.01, 1.01, 1.01)
                selectActor.GetProperty().SetPointSize(3)

                # self.vtkWidget.viewer.getRenderer().AddActor(actor)
                self.vtkWidget.viewer.getRenderer().AddActor(selectActor)
                # self.vtkWidget.viewer.getRenderer().AddActor(sphere_actor)
                print("currently present actors",
                      self.vtkWidget.viewer.getRenderer().GetActors().GetNumberOfItems())
                self.pointActorsAdded = True
        else:
            print("pointcloud already added")

    def updateClippingPlane(self, obj, event):
        print("caught updateClippingPlane!", event)
        normal = [0, 0, 0]
        origin = [0, 0, 0]
        orientation = self.vtkWidget.viewer.GetSliceOrientation()
        norm = 1
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

        normal[orientation] = norm
        origin[orientation] = self.vtkWidget.viewer.GetActiveSlice() + beta
        print("normal", normal)
        print("origin", origin)
        print("<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>")

        self.visPlane.SetOrigin(origin[0], origin[1], origin[2])
        self.visPlane.SetNormal(normal[0], normal[1], normal[2])

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

        self.tableDock.hide()

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

            vox = self.vtkWidget.viewer.style.display2imageCoordinate(position)
            print("[%d,%d,%d] : %.2g" % vox)
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
                el = 2
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

    def appendPointToCloud(self, point):
        self.pointcloud.append(point)
        print(self.pointcloud)
        p = self.vtkPointCloud.InsertNextPoint(self.pointcloud[-1][1],
                                               self.pointcloud[-1][2],
                                               self.pointcloud[-1][3])
        self.vertices.InsertNextCell(1)
        self.vertices.InsertCellPoint(p)


def main():
    err = vtk.vtkFileOutputWindow()
    err.SetFileName("viewer.log")
    vtk.vtkOutputWindow.SetInstance(err)

    App = QApplication(sys.argv)
    gui = Window()
    if len(sys.argv) > 1:
        gui.setSubvolSize(float(sys.argv[1]))
        gui.dislayPointCloudAsSpheres(False)
    sys.exit(App.exec())


if __name__ == "__main__":
    main()
