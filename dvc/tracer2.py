# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:48:22 2018

@author: ofn77899
"""

#!/usr/bin/env python

# initial translation from the tcl by VTK/Utilities/tcl2py.py
# further cleanup and fixes to the translation by Charl P. Botha

import vtk
from vtk.util.misc import vtkGetDataRoot
VTK_DATA_ROOT = vtkGetDataRoot()
from ccpi.viewer.CILViewer2D import CILViewer2D
from ccpi.viewer.CILViewer2D import Converter
from ccpi.viewer.CILViewer import CILViewer
import os
import numpy


def createPoints(density , sliceno, image_data, orientation ):
    '''creates a 2D point cloud on the image data on the selected orientation
    
    input:
        density: points/voxel (list or tuple)
        image_data: vtkImageData onto project the pointcloud
        orientation: orientation of the slice onto which create the point cloud
        
    returns: 
        vtkPoints
    '''
    vtkPointCloud = vtk.vtkPoints()
    image_spacing = list ( image_data.GetSpacing() ) 
    image_origin  = list ( image_data.GetOrigin() )
    image_dimensions = list ( image_data.GetDimensions() )
    print ("spacing    : ", image_spacing)
    print ("origin     : ", image_origin)
    print ("dimensions : ", image_dimensions)
    # reduce to 2D on the proper orientation
    spacing_z = image_spacing.pop(orientation)
    origin_z  = image_origin.pop(orientation)
    dim_z     = image_dimensions.pop(orientation)
       
    # the total number of points on X and Y axis
    max_x = int(image_dimensions[0] * density[0] )
    max_y = int(image_dimensions[1] * density[1] )
    
    print ("max_x: {} {} {}".format(max_x, image_dimensions, density))
    print ("max_y: {} {} {}".format(max_y, image_dimensions, density))
    
    z = sliceno * spacing_z - origin_z
    print ("Sliceno {} Z {}".format(sliceno, z))
    
    # skip the offset in voxels
    offset = [1,1]
    n_x = offset[0]
     
    while n_x < max_x:
        # x axis
        n_y = offset[1]
        while n_y < max_y:
            # y axis
            x = (n_x / max_x) * image_spacing[0] * image_dimensions[0]- image_origin[0] #+ int(image_dimensions[0] * density[0] * .7) 
            y = (n_y / max_y) * image_spacing[1] * image_dimensions[1]- image_origin[1] #+ int(image_dimensions[1] * density[1] * .7)
            
            p = vtkPointCloud.InsertNextPoint( x , y , z)
            
            
            n_y += 1
            
        n_x += 1
    
    return vtkPointCloud  

def points2vertices(points):
    '''returns a vtkCellArray from a vtkPoints'''
    
    vertices = vtk.vtkCellArray()
    for i in range(points.GetNumberOfPoints()):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
        # print (points.GetPoint(i))
    return vertices

def world2imageCoordinate(world_coordinates, imagedata):
    """
    Convert from the world or global coordinates to image coordinates
    :param world_coordinates: (x,y,z)
    :return: rounded to next integer (x,y,z) in image coorindates eg. slice index
    """
    # dims = imagedata.GetDimensions()
    spac = imagedata.GetSpacing()
    orig = imagedata.GetOrigin()

    return [round(world_coordinates[i] / spac[i] + orig[i]) for i in range(3)]
    
class vtkMaskPolyData():
    def __init__(self):
        self.polydata = None
        self.mask = None
        self.mask_value = 1
        self.point_in_mask = 0
        
    def SetPolyDataInput(self, polydata):
        self.polydata = polydata
    def SetMask(self, mask):
        self.mask = mask
    def SetMaskValue(self, value):
        self.mask_value = value
    def GetOutput(self):
        '''returns a polydata object'''
        self.point_in_mask = 0
        if self.polydata and self.mask:
            in_points = self.polydata.GetPoints()
            out_points = vtk.vtkPoints()
            for i in range(in_points.GetNumberOfPoints()):
                pp = in_points.GetPoint(i)
                
                # get the point in image coordinate
                
                ic = world2imageCoordinate(pp, self.mask)
                i = 0
                outside = False
                while i < len(ic):
                    outside = ic[i] < 0 or ic[i] >= self.mask.GetDimensions()[i]
                    if outside:
                        break
                    i += 1

                if not outside:
                    mm = self.mask.GetScalarComponentAsDouble(int(ic[0]), 
                                                          int(ic[1]),
                                                          int(ic[2]), 0)
                    print ("value of point {} {}".format(mm, ic))
                    if mm == self.mask_value:
                        out_points.InsertNextPoint(*pp)
                        self.point_in_mask += 1
            
            vertices = points2vertices(out_points)
            pointPolyData = vtk.vtkPolyData()
            pointPolyData.SetPoints(out_points)
            pointPolyData.SetVerts(vertices)
            return pointPolyData
    

err = vtk.vtkFileOutputWindow()
err.SetFileName("tracer2.log")
vtk.vtkOutputWindow.SetInstance(err)
    
# Start by loading some data.
v16 = vtk.vtkMetaImageReader()
v16.SetFileName(os.path.abspath("../../CCPi-Simpleflex/data/head.mha"))
v16.Update()




path = []
path.append( (74.66533660888672, 127.88501739501953, 0.0) )
path.append( (76.09085845947266, 121.23255920410156, 0.0) )
path.append( (76.09085845947266, 114.10491943359375, 0.0) )
path.append( (76.5660400390625, 111.25386810302734, 0.0) )
path.append( (77.04121398925781, 110.30352020263672, 0.0) )
path.append( (81.31779479980469, 104.12622833251953, 0.0) )
path.append( (81.79296875, 104.12622833251953, 0.0) )
path.append( (81.79296875, 103.65105438232422, 0.0) )
path.append( (82.26815032958984, 103.65105438232422, 0.0) )
path.append( (82.74332427978516, 103.1758804321289, 0.0) )
path.append( (85.59437561035156, 104.12622833251953, 0.0) )
path.append( (87.49507904052734, 106.50211334228516, 0.0) )
path.append( (89.87095642089844, 108.40281677246094, 0.0) )
path.append( (92.72201538085938, 110.30352020263672, 0.0) )
path.append( (95.57306671142578, 112.67939758300781, 0.0) )
path.append( (98.42412567138672, 114.5801010131836, 0.0) )
path.append( (98.89929962158203, 115.0552749633789, 0.0) )
path.append( (103.1758804321289, 119.8070297241211, 0.0) )
path.append( (103.65105438232422, 121.70773315429688, 0.0) )
path.append( (104.60140991210938, 123.13326263427734, 0.0) )
path.append( (105.07658386230469, 126.9346694946289, 0.0) )
path.append( (105.07658386230469, 128.36019897460938, 0.0) )
path.append( (105.07658386230469, 128.8353729248047, 0.0) )
path.append( (105.07658386230469, 129.310546875, 0.0) )
path.append( (105.07658386230469, 129.7857208251953, 0.0) )
path.append( (104.12622833251953, 129.7857208251953, 0.0) )
path.append( (100.80000305175781, 129.7857208251953, 0.0) )
path.append( (95.57306671142578, 129.7857208251953, 0.0) )
path.append( (93.19718933105469, 129.7857208251953, 0.0) )
path.append( (92.72201538085938, 129.7857208251953, 0.0) )
path.append( (92.24684143066406, 129.7857208251953, 0.0) )
path.append( (91.77165985107422, 129.7857208251953, 0.0) )
path.append( (89.87095642089844, 129.310546875, 0.0) )
path.append( (88.92060852050781, 129.310546875, 0.0) )
path.append( (88.4454345703125, 129.310546875, 0.0) )
path.append( (86.54473114013672, 129.310546875, 0.0) )
path.append( (86.06954956054688, 129.310546875, 0.0) )
path.append( (85.59437561035156, 129.310546875, 0.0) )
path.append( (85.11920166015625, 129.310546875, 0.0) )
path.append( (85.59437561035156, 129.7857208251953, 0.0) )

pathpoints = vtk.vtkPoints()
for p in path:
    pathpoints.InsertNextPoint(p[0],p[1],3 * v16.GetOutput().GetSpacing()[2])

# create a blank image
dims = v16.GetOutput().GetDimensions()
mask0 = Converter.numpy2vtkImporter(numpy.zeros(
                                     (dims[2],dims[1],dims[0]), 
                                     order='F', dtype=numpy.uint16), 
                                   spacing = v16.GetOutput().GetSpacing(), 
                                   origin = v16.GetOutput().GetOrigin(),
                                   transpose=[0,1,2]
                                   )
mask1 = Converter.numpy2vtkImporter(numpy.ones(
                                     (dims[2],dims[1],dims[0]), 
                                     order='F', dtype=numpy.uint16), 
                                   spacing = v16.GetOutput().GetSpacing(), 
                                   origin = v16.GetOutput().GetOrigin(),
                                   transpose=[0,1,2]
                                   )
mask0.Update()
mask1.Update()
print ("mask0", mask0.GetOutput().GetScalarTypeAsString())
print ("mask1", mask1.GetOutput().GetScalarTypeAsString())
print ("image", v16.GetOutput().GetScalarTypeAsString())

lasso = vtk.vtkLassoStencilSource()
lasso.SetShapeToPolygon()
lasso.SetSlicePoints(0,pathpoints)
lasso.SetInformationInput(v16.GetOutput())

stencil = vtk.vtkImageStencil()
stencil.SetInputConnection(mask1.GetOutputPort())
stencil.SetBackgroundInputData(mask0.GetOutput())
stencil.SetStencilConnection(lasso.GetOutputPort())
stencil.Update()

# create points in mask
# 1 point per voxel both in x and y
density = (1,2) 
# which slice
sliceno = 0
orientation = 2
origin = v16.GetOutput().GetOrigin()
spacing = v16.GetOutput().GetSpacing()
dimensions = v16.GetOutput().GetDimensions()

rotate = (0,30.,0)
transform = vtk.vtkTransform()
transform.Translate(dimensions[0]/2*spacing[0], dimensions[1]/2*spacing[1],0)
transform.RotateZ(30.)
transform.Translate(-dimensions[0]/2*spacing[0], -dimensions[1]/2*spacing[1],0)

points = createPoints(density, sliceno , v16.GetOutput(), orientation)

#%%




vertices = points2vertices(points)


pointPolyData = vtk.vtkPolyData()
pointPolyData.SetPoints(points)
pointPolyData.SetVerts(vertices)
     
t_filter = vtk.vtkTransformFilter()
t_filter.SetTransform(transform)
t_filter.SetInputData(pointPolyData)
t_filter.Update()


masked_polydata = vtkMaskPolyData()
masked_polydata.SetMask(stencil.GetOutput())
masked_polydata.SetPolyDataInput(t_filter.GetOutput())
print ("masked point", masked_polydata.point_in_mask)


mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputConnection(t_filter.GetOutputPort())
mapper.SetInputData(masked_polydata.GetOutput())

actor = vtk.vtkLODActor()
actor.SetMapper(mapper)
actor.GetProperty().SetPointSize(3)
actor.GetProperty().SetColor(1, .2, .2)
actor.VisibilityOn()



v = CILViewer2D()
# v.setInput3DData(v16.GetOutput())
v.setInput3DData(stencil.GetOutput())
v.ren.AddActor(actor)
v.startRenderLoop()