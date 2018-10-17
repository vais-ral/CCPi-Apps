# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:32:00 2018

@author: ofn77899
"""

from ccpi.viewer.CILViewer2D import CILViewer2D
from ccpi.viewer.CILViewer2D import Converter
from ccpi.viewer.CILViewer import CILViewer
import vtk
import os
import numpy
import csv
import json
from functools import reduce

def loadPointCloudFromCSV(filename, delimiter=','):
    print ("loadPointCloudFromCSV")
    pointcloud = []
    with open(filename, 'r') as csvfile:
        read = csv.reader(csvfile, delimiter=delimiter)
        for row in read:
            #read in only numerical values
            print (row)
            try:
                row = list(map(lambda x: float(x),row))
            #print ("reduce " , reduce( lambda x,y: isinstance(x,Number) and \
            #          isinstance(y,Number) , row))
            #if reduce( lambda x,y: isinstance(x,Number) and \
            #          isinstance(y,Number) , row):
                pointcloud.append(row)
            except ValueError as ve:
                print ('Value Error' , ve)
    return pointcloud

displ = numpy.asarray(
        loadPointCloudFromCSV('fracture.disp','\t')[:]
        )

#displ[10][6] = 20.
#displ[10][7] = 0.
#displ[10][8] = 0.
#displ[11][6] = 0.
#displ[11][7] = 20.
#displ[11][8] = 0.
#displ[12][6] = 0.
#displ[12][7] = 0.
#displ[12][8] = 20.

dist = (displ.T[6]**2 + displ.T[7]**2 + displ.T[8]**2)
m = dist.min()
M = dist.max()
#%%


grid = vtk.vtkUnstructuredGrid()
vertices = vtk.vtkCellArray()
arrow = vtk.vtkDoubleArray()
arrow.SetNumberOfComponents(3)
acolor = vtk.vtkDoubleArray()

pc = vtk.vtkPoints()
for count in range(len(displ)):
    p = pc.InsertNextPoint(displ[count][1],
                           displ[count][2], 
                           displ[count][3])
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(p)
    #arrow.InsertNextTuple3(displ[count][6],displ[count][7],displ[count][8])
    arrow.InsertNextTuple3(displ[count][6],displ[count][7],0)
    acolor.InsertNextValue(reduce(lambda x,y: x + y**2, (*displ[count][6:8],0), 0))
    #(displ[count][6]**2+displ[count][7]**2+displ[count][8]**2))
    
lut = vtk.vtkLookupTable()
print ("lut table range" , acolor.GetRange())
lut.SetTableRange(acolor.GetRange())
lut.SetNumberOfTableValues( 256 )
lut.SetHueRange( 240/360., 0. )
#lut.SetSaturationRange( 1, 1 )
lut.Build()

pointPolyData = vtk.vtkPolyData()
#2. Add the points to a vtkPolyData.
pointPolyData.SetPoints( pc ) 
pointPolyData.SetVerts( vertices ) 
pointPolyData.GetPointData().SetVectors(arrow)
pointPolyData.GetPointData().SetScalars(acolor)

# arrow
arrow_glyph = vtk.vtkGlyph3D()
#arrow_glyph.SetScaleModeToDataScalingOff()
arrow_glyph.SetScaleModeToScaleByVector()
#arrow_glyph.SetColorModeToColorByVector()
arrow_source = vtk.vtkArrowSource()
arrow_source.SetTipRadius(0.2)
arrow_source.SetShaftRadius(0.075)
arrow_mapper = vtk.vtkPolyDataMapper()
arrow_mapper.SetInputConnection(arrow_glyph.GetOutputPort())
arrow_mapper.SetScalarModeToUsePointFieldData()
arrow_mapper.SelectColorArray(0)
arrow_mapper.SetScalarRange(acolor.GetRange())
arrow_mapper.SetLookupTable(lut)

arrow_glyph.SetInputData(pointPolyData)
arrow_glyph.SetSourceConnection(arrow_source.GetOutputPort())
#arrow_glyph.SetScaleFactor(5)
arrow_glyph.SetScaleModeToScaleByVector()
arrow_glyph.SetVectorModeToUseVector()
arrow_glyph.ScalingOn()
arrow_glyph.OrientOn()

# Usual actor
arrow_actor = vtk.vtkActor()
arrow_actor.SetMapper(arrow_mapper)
#arrow_actor.GetProperty().SetColor(0, 1, 1)

# vtk user guide p.95
conesource = vtk.vtkConeSource()
conesource.SetResolution(6)
transform = vtk.vtkTransform()
transform.Translate(0.5,0.,0.)
transformF = vtk.vtkTransformPolyDataFilter()
transformF.SetInputConnection(conesource.GetOutputPort())
transformF.SetTransform(transform)

cones = vtk.vtkGlyph3D()
cones.SetInputData(pointPolyData)
cones.SetSourceConnection(transformF.GetOutputPort())


mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(cones.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)
#actor.GetProperty().SetPointSize(3)
actor.GetProperty().SetColor(0,1,1)

pmapper = vtk.vtkPolyDataMapper()
pmapper.SetInputData(pointPolyData)

pactor = vtk.vtkActor()
pactor.SetMapper(pmapper)
pactor.GetProperty().SetPointSize(3)
pactor.GetProperty().SetColor(1,0,1)

v = CILViewer()
v.ren.AddActor(pactor)
#v.ren.AddActor(actor)
v.ren.AddActor(arrow_actor)


## add volume
if True:
    runconfig = json.load(open("dvc_kalpani.json", "r"))
    dataset = numpy.load(os.path.abspath(runconfig['correlate_filename']))
    conv = Converter.numpy2vtkImporter(numpy.transpose(dataset, [0,1,2]))
    conv.Update()
    v.setInput3DData(conv.GetOutput())
    v.style.SetActiveSlice(255)
    v.style.UpdatePipeline()
    
v.startRenderLoop()