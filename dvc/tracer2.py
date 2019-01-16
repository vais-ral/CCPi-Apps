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
from numbers import Integral, Number
from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase


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
    # print ("spacing    : ", image_spacing)
    # print ("origin     : ", image_origin)
    # print ("dimensions : ", image_dimensions)
    # reduce to 2D on the proper orientation
    spacing_z = image_spacing.pop(orientation)
    origin_z  = image_origin.pop(orientation)
    dim_z     = image_dimensions.pop(orientation)
       
    # the total number of points on X and Y axis
    max_x = int(image_dimensions[0] * density[0] )
    max_y = int(image_dimensions[1] * density[1] )
    
    # print ("max_x: {} {} {}".format(max_x, image_dimensions, density))
    # print ("max_y: {} {} {}".format(max_y, image_dimensions, density))
    
    z = sliceno * spacing_z - origin_z
    # print ("Sliceno {} Z {}".format(sliceno, z))
    
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

class cilMaskPolyDataV1(object):
    '''A filter to mask a vtkPolyData with an vtkImageData
    
    https://blog.kitware.com/vtkpythonalgorithm-is-great/
    '''
    def __init__(self):
        self._mask_value = 0
        super(cilMaskPolyData, self).__init__()
    
    @property
    def mask_value(self):
        return self._mask_value
    @mask_value.setter
    def mask_value(self, mask_value):
        if not isinstance(mask_value, Integral):
            raise ValueError('Mask value must be an integer. Got' , mask_value)
        self._mask_value = mask_value
        
    # VTK specific pipeline    
    def Initialize(self, vtkself):
        '''initialize the number of input and output ports of the algorithm'''
        vtkself.SetNumberOfInputPorts(1)
        vtkself.SetNumberOfOutputPorts(1)
    def FillInputPortInformation(self, vtkself, port, info):
        info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet")
        return 1
 
    def FillOutputPortInformation(self, vtkself, port, info):
        info.Set(vtk.vtkDataObject.DATA_TYPE_NAME(), "vtkPolyData")
        return 1
    
    def ProcessRequest(self, vtkself, request, inInfo, outInfo):
        if request.Has(vtk.vtkDemandDrivenPipeline.REQUEST_DATA()):
            inp = inInfo[0].GetInformationObject(0).Get(vtk.vtkDataObject.DATA_OBJECT())
            opt = outInfo.GetInformationObject(0).Get(vtk.vtkDataObject.DATA_OBJECT())
 
            cf = vtk.vtkContourFilter()
            cf.SetInputData(inp)
            cf.SetValue(0, self.mask_value)
 
            sf = vtk.vtkShrinkPolyData()
            sf.SetInputConnection(cf.GetOutputPort())
            sf.Update()
 
            opt.ShallowCopy(sf.GetOutput())
        return 1
    
    
class vtkMaskPolyData(object):
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
                    
                    if int(mm) == self.mask_value:
                        print ("value of point {} {}".format(mm, ic))
                        out_points.InsertNextPoint(*pp)
                        self.point_in_mask += 1
            
            vertices = points2vertices(out_points)
            pointPolyData = vtk.vtkPolyData()
            pointPolyData.SetPoints(out_points)
            pointPolyData.SetVerts(vertices)
            return pointPolyData

class cilMaskPolyData(VTKPythonAlgorithmBase):
    def __init__(self):
          VTKPythonAlgorithmBase.__init__(self, nInputPorts=2, nOutputPorts=1)
          self.__MaskValue = 1
          
    def SetMaskValue(self, mask_value):
        if not isinstance(mask_value, Integral):
            raise ValueError('Mask value must be an integer. Got' , mask_value)
        
        if mask_value != self.__MaskValue:
            self.__MaskValue = mask_value
            self.Modified()
  
    def GetMaskValue(self):
        return self.__MaskValue
            
    def FillInputPortInformation(self, port, info):
        if port == 0:
            info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData")
        elif port == 1:
            info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkImageData")
        return 1
 
    def FillOutputPortInformation(self, port, info):
        info.Set(vtk.vtkDataObject.DATA_TYPE_NAME(), "vtkPolyData")
        return 1
  
    def RequestData(self, request, inInfo, outInfo):
        self.point_in_mask = 0
        in_points = vtk.vtkDataSet.GetData(inInfo[0])
        mask = vtk.vtkDataSet.GetData(inInfo[1])
        out_points = vtk.vtkPoints()
        for i in range(in_points.GetNumberOfPoints()):
            pp = in_points.GetPoint(i)
            
            # get the point in image coordinate
            
            ic = self.world2imageCoordinate(pp, mask)
            i = 0
            outside = False
            while i < len(ic):
                outside = ic[i] < 0 or ic[i] >= mask.GetDimensions()[i]
                if outside:
                    break
                i += 1

            if not outside:
                mm = mask.GetScalarComponentAsDouble(int(ic[0]), 
                                                      int(ic[1]),
                                                      int(ic[2]), 0)
                
                if int(mm) == self.GetMaskValue():
                    print ("value of point {} {}".format(mm, ic))
                    out_points.InsertNextPoint(*pp)
                    self.point_in_mask += 1
        
        vertices = self.points2vertices(out_points)
        pointPolyData = vtk.vtkPolyData.GetData(outInfo)
        pointPolyData.SetPoints(out_points)
        pointPolyData.SetVerts(vertices)
        print ("points in mask", self.point_in_mask)
        return 1
    
    def world2imageCoordinate(self, world_coordinates, imagedata):
        """
        Convert from the world or global coordinates to image coordinates
        :param world_coordinates: (x,y,z)
        :return: rounded to next integer (x,y,z) in image coorindates eg. slice index
        """
        # dims = imagedata.GetDimensions()
        spac = imagedata.GetSpacing()
        orig = imagedata.GetOrigin()
    
        return [round(world_coordinates[i] / spac[i] + orig[i]) for i in range(3)]
    def points2vertices(self, points):
        '''returns a vtkCellArray from a vtkPoints'''
        
        vertices = vtk.vtkCellArray()
        for i in range(points.GetNumberOfPoints()):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
            # print (points.GetPoint(i))
        return vertices

class cilClipPolyDataBetweenPlanes(VTKPythonAlgorithmBase):
    def __init__(self):
          VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1)
          self.__PlaneOrigin    = 0
          self.__PlaneNormal    = 1
          self.__PlaneDistance  = 1
          self.planesource = [ vtk.vtkPlaneSource(), vtk.vtkPlaneSource() ]
          self.visPlane = [ vtk.vtkPlane() , vtk.vtkPlane() ]
          self.planeClipper =  [ vtk.vtkClipPolyData() , vtk.vtkClipPolyData()]
          
          self.planesource[0].SetCenter(self.visPlane[0].GetOrigin())
          self.planesource[1].SetCenter(self.visPlane[1].GetOrigin())
          self.planesource[0].SetNormal(self.visPlane[0].GetNormal())
          self.planesource[1].SetNormal(self.visPlane[1].GetNormal())
          
    def SetPlaneOrigin(self, value):
        if not isinstance(value, Number):
            raise ValueError('PlaneOrigin must be a number. Got' , value)
        
        if value != self.__PlaneOrigin:
            self.__PlaneOrigin = value
            self.Modified()
  
    def GetPlaneOrigin(self):
        return self.__PlaneOrigin
    
    def SetPlaneNormal(self, value):
        if not isinstance(value, Number):
            raise ValueError('PlaneNormal must be a number. Got' , value)
        
        if value != self.__PlaneNormal:
            self.__PlaneNormal = value
            self.Modified()
  
    def GetPlaneNormal(self):
        return self.__PlaneNormal
    
    def SetPlaneDistance(self, value):
        if not isinstance(value, Number):
            raise ValueError('PlaneDistance must be a number. Got' , value)
        if not value > 0:
            raise ValueError('PlaneDistance must be positive.')
        
        if value != self.__PlaneDistance:
            self.__PlaneDistance = value
            self.Modified()
  
    def GetPlanePlaneDistance(self):
        return self.__PlaneDistance
            
    def FillInputPortInformation(self, port, info):
        if port == 0:
            info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkPolyData")
        
        return 1
 
    def FillOutputPortInformation(self, port, info):
        info.Set(vtk.vtkDataObject.DATA_TYPE_NAME(), "vtkPolyData")
        return 1
  
    def RequestData(self, request, inInfo, outInfo):
        
        return 1
    
#%%
class cilRegularPointCloudToPolyData(VTKPythonAlgorithmBase):
    CIRCLE = 'circle'
    SQUARE = 'square'
    CUBE   = 'cube'
    SPHERE = 'sphere'
    def __init__(self):
          VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1)
          self.__Points = vtk.vtkPoints()
          self.__Vertices = vtk.vtkCellArray()
          self.__Density = [ 1., 1., 1.]
          self.__Orientation = 2
          self.__Overlap = [0.2, 0.2, 0.2] #: 3D overlap
          self.__Dimensionality = 3
          self.__SliceNumber = 0
          self.__Mode = self.SPHERE
          self.__SubVolumeRadius = 1 #: Radius of the subvolume in voxels
          
    def SetMode(self, value):
        if not value in [self.CIRCLE, self.SQUARE, self.CUBE, self.SPHERE]:
            raise ValueError('dimension must be in [circle, square, cube, sphere]. Got', 
                             value)
        
        if value != self.__Mode:
            self.__Mode = value
            self.Modified()
  
    def GetMode(self):
        return self.__Mode
    
    def SetDensity(self, dimension, value):
        if not isinstance(value, Number):
            raise ValueError('Mask value must be a number. Got' , value)
        if not dimension in [0, 1, 2]:
            raise ValueError('dimension must be in [0, 1, 2]. Got' , value)
        
        
        if value != self.__Density[dimension]:
            self.__Density[dimension] = value
            self.Modified()
  
    def GetDensity(self):
        return self.__Density
    
    def SetDimensionality(self, value):
        if not value in [2, 3]:
            raise ValueError('Dimensionality must be in [2, 3]. Got', value)
        
        if self.__Dimensionality != value:
            self.__Dimensionality = value
            self.Modified()
    def GetDimensionality(self):
        return self.__Dimensionality
    
    def SetOverlap(self, dimension, value):
        if not isinstance(value, Number):
            raise ValueError('Overlap value must be a number. Got' , value)
        if not dimension in [0, 1, 2]:
            raise ValueError('dimension must be in [0, 1, 2]. Got' , value)
        if value != self.__Overlap[dimension]:
            self.__Overlap[dimension] = value
            self.Modified()
    def GetOverlap(self):
        return self.__Overlap
    
    def SetSlice(self, value):
        if not isinstance(value, int):
            raise ValueError('Slice must be a positive integer. Got', value)
        if not value >= 0:
            raise ValueError('Slice must be a positive integer. Got', value)
        if self.__SliceNumber != value:
            self.__SliceNumber = value
            self.Modified()
    def GetSlice(self):
        return self.__SliceNumber
        
    def GetNumberOfPoints(self):
        return self.__Points.GetNumberOfPoints()
    
    def SetOrientation(self, value):
        if not value in [0, 1, 2]:
            raise ValueError('Orientation must be in [0,1,2]. Got', value)
        if self.__Orientation != value:
            self.__Orientation = value
            self.Modified()
    
    def GetOrientation(self):
        return self.__Orientation
    
    def SetSubVolumeRadiusInVoxel(self, value):
        '''Set the radius of the subvolume in voxel'''
        if not isinstance(value, Integral):
            raise ValueError('SubVolumeRadius must be an integer larger than 1. Got', value)
        if not value > 1:
            raise ValueError('SubVolumeRadius must be an integer larger than 1. Got', value)
        if self.__SubVolumeRadius != value:
            self.__SubVolumeRadius = value
            self.Modified()
    
    def GetSubVolumeRadiusInVoxel(self):
        return self.__SubVolumeRadius
            
    def FillInputPortInformation(self, port, info):
        if port == 0:
            info.Set(vtk.vtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "vtkImageData")
        return 1
 
    def FillOutputPortInformation(self, port, info):
        info.Set(vtk.vtkDataObject.DATA_TYPE_NAME(), "vtkPolyData")
        return 1
  
    def RequestData(self, request, inInfo, outInfo):
        
        print ("Request Data")
        image_data = vtk.vtkDataSet.GetData(inInfo[0])
        pointPolyData = vtk.vtkPolyData.GetData(outInfo)
        
        orientation = self.GetOrientation()
        print ("orientation", orientation)
        dimensionality = self.GetDimensionality()
        print ("dimensionality", dimensionality)
        
        overlap = self.GetOverlap()
        print ("overlap", overlap)
        point_spacing = self.CalculatePointSpacing(overlap, mode=self.GetMode())
        print ("density", density)
        if dimensionality == 3:
            self.CreatePoints3D(point_spacing, image_data, orientation)
        else:
            sliceno = self.GetSlice()
            if image_data.GetDimensions()[orientation] < sliceno:
                raise ValueError('Requested slice is outside the image.' , sliceno)
            
            self.CreatePoints2D(point_spacing, sliceno, image_data, orientation)
            
        self.FillCells()
        
        pointPolyData.SetPoints(self.__Points)
        pointPolyData.SetVerts(self.__Vertices)
        return 1
    
    def CreatePoints2D(self, point_spacing , sliceno, image_data, orientation ):
        '''creates a 2D point cloud on the image data on the selected orientation
        
        input:
            density: points/voxel (list or tuple)
            image_data: vtkImageData onto project the pointcloud
            orientation: orientation of the slice onto which create the point cloud
            
        returns: 
            vtkPoints
        '''
        vtkPointCloud = self.__Points
        image_spacing = list ( image_data.GetSpacing() ) 
        image_origin  = list ( image_data.GetOrigin() )
        image_dimensions = list ( image_data.GetDimensions() )
        # print ("spacing    : ", image_spacing)
        # print ("origin     : ", image_origin)
        # print ("dimensions : ", image_dimensions)
        # reduce to 2D on the proper orientation
        spacing_z = image_spacing.pop(orientation)
        origin_z  = image_origin.pop(orientation)
        dim_z     = image_dimensions.pop(orientation)
           
        # the total number of points on X and Y axis
        max_x = int(image_dimensions[0] / point_spacing[0] )
        max_y = int(image_dimensions[1] / point_spacing[1] )
        
        # print ("max_x: {} {} {}".format(max_x, image_dimensions, density))
        # print ("max_y: {} {} {}".format(max_y, image_dimensions, density))
        
        z = sliceno * spacing_z - origin_z
        # print ("Sliceno {} Z {}".format(sliceno, z))
        
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
    
    def CreatePoints3D(self, point_spacing , image_data, orientation ):
        '''creates a 2D point cloud on the image data on the selected orientation
        
        input:
            density: points/voxel (list or tuple)
            image_data: vtkImageData onto project the pointcloud
            orientation: orientation of the slice onto which create the point cloud
            
        returns: 
            vtkPoints
        '''
        vtkPointCloud = self.__Points
        image_spacing = list ( image_data.GetSpacing() ) 
        image_origin  = list ( image_data.GetOrigin() )
        image_dimensions = list ( image_data.GetDimensions() )
          
        # the total number of points on X and Y axis
        max_x = int(image_dimensions[0] / point_spacing[0] )
        max_y = int(image_dimensions[1] / point_spacing[1] )
        max_z = int(image_dimensions[2] / point_spacing[2] )
        
        # print ("max_x: {} {} {}".format(max_x, image_dimensions, density))
        # print ("max_y: {} {} {}".format(max_y, image_dimensions, density))
        
        # print ("Sliceno {} Z {}".format(sliceno, z))
        
        # skip the offset in voxels
        # radius = self.GetSubVolumeRadiusInVoxel()
        offset = [0, 0, 0]
        n_x = offset[0]
         
        while n_x < max_x:
            # x axis
            n_y = offset[1]
            while n_y < max_y:
                # y axis
                n_z = offset[1]
                while n_z < max_z:
                    x = (n_x / max_x) * image_spacing[0] * image_dimensions[0]- image_origin[0] #+ int(image_dimensions[0] * density[0] * .7) 
                    y = (n_y / max_y) * image_spacing[1] * image_dimensions[1]- image_origin[1] #+ int(image_dimensions[1] * density[1] * .7)
                    z = (n_z / max_z) * image_spacing[2] * image_dimensions[2]- image_origin[2] #+ int(image_dimensions[1] * density[1] * .7)
                    
                    p = vtkPointCloud.InsertNextPoint( x, y, z )
                    n_z += 1
                
                n_y += 1
                
            n_x += 1
        
        return 1
    def FillCells(self):
        vertices = self.__Vertices
        number_of_cells = vertices.GetNumberOfCells()
        for i in range(self.GetNumberOfPoints()):
            if i >= number_of_cells:
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(i)
    
    def CalculatePointSpacing(self, overlap, mode=SPHERE):
        '''returns the ratio between the figure size (radius) and the distance between 2 figures centers in 3D'''
        print ("CalculateDensity", overlap)
        
        if isinstance (overlap, tuple) or isinstance(overlap, list):
            d = [self.distance_from_overlap(ovl, mode=mode) for ovl in overlap]
        elif isinstance(overlap, float):
            d = [self.distance_from_overlap(overlap, mode=mode)]
            d += [d[-1]]
            d += [d[-1]]
        return d
        
            
    def overlap(self, radius, center_distance, mode=SPHERE):
        if center_distance <= 2*radius:
            if mode == 'circle':
                overlap = (2 * numpy.acos(center_distance/radius/2.) - \
                           (center_distance/radius) *  numpy.sqrt(1 - \
                           (center_distance/radius/2.)*(center_distance/radius/2.)) \
                          ) / 3.1415 
            elif mode == 'square':
                overlap = (1 - center_distance/radius ) 
            elif mode == 'cube':
                overlap = (1 - center_distance/radius ) 
            elif mode == 'sphere':
                overlap = (2. * radius - center_distance)**2  *\
                          (center_distance + 4 * radius) / \
                          (16 * radius ** 3 ) 
            else:
                raise ValueError('unsupported mode',mode)
        else:
            overlap = 0
        return overlap
    
    def distance_from_overlap(self, req, interp=False, N=1000, mode='sphere'):
        '''hard inversion'''
        radius = self.GetSubVolumeRadiusInVoxel()
        x = [2.* i/N * radius for i in range(N+1)]
        y = [self.overlap(radius, x[i], mode=mode) - req for i in range(N+1)]
        # find the value closer to 0 for required overlap
        idx = (y.index(min (y, key=abs)))
        if interp:
            if y[idx] * y[idx+1] < 0:
                m = (y[idx] -y[idx+1]) / (x[idx] -x[idx+1])
            else:
                m = (y[idx] -y[idx-1]) / (x[idx] -x[idx-1])
            q = y[idx] - m * x[idx]
            x0 = -q / m
        else:
            x0 = x[idx]
        return x0
#%%
        
    
err = vtk.vtkFileOutputWindow()
err.SetFileName("tracer2.log")
vtk.vtkOutputWindow.SetInstance(err)
    
# Start by loading some data.
v16 = vtk.vtkMetaImageReader()
v16.SetFileName(os.path.abspath("../../CCPi-Simpleflex/data/head.mha"))
v16.Update()
origin = v16.GetOutput().GetOrigin()
spacing = v16.GetOutput().GetSpacing()
dimensions = v16.GetOutput().GetDimensions()

# create points in mask
# 1 point per voxel both in x and y
density = (0.5,0.6) 
# which slice
sliceno = 3
orientation = 2



path = []
path.append( (74.66533660888672, 127.88501739501953, 0.0) )
#path.append( (76.09085845947266, 121.23255920410156, 0.0) )
#path.append( (76.09085845947266, 114.10491943359375, 0.0) )
#path.append( (76.5660400390625, 111.25386810302734, 0.0) )
path.append( (77.04121398925781, 110.30352020263672, 0.0) )
#path.append( (81.31779479980469, 104.12622833251953, 0.0) )
#path.append( (81.79296875, 104.12622833251953, 0.0) )
#path.append( (81.79296875, 103.65105438232422, 0.0) )
path.append( (82.26815032958984, 103.65105438232422, 0.0) )
#path.append( (82.74332427978516, 103.1758804321289, 0.0) )
#path.append( (85.59437561035156, 104.12622833251953, 0.0) )
#path.append( (87.49507904052734, 106.50211334228516, 0.0) )
path.append( (89.87095642089844, 108.40281677246094, 0.0) )
#path.append( (92.72201538085938, 110.30352020263672, 0.0) )
#path.append( (95.57306671142578, 112.67939758300781, 0.0) )
#path.append( (98.42412567138672, 114.5801010131836, 0.0) )
path.append( (98.89929962158203, 115.0552749633789, 0.0) )
#path.append( (103.1758804321289, 119.8070297241211, 0.0) )
#path.append( (103.65105438232422, 121.70773315429688, 0.0) )
#path.append( (104.60140991210938, 123.13326263427734, 0.0) )
path.append( (105.07658386230469, 126.9346694946289, 0.0) )
#path.append( (105.07658386230469, 128.36019897460938, 0.0) )
#path.append( (105.07658386230469, 128.8353729248047, 0.0) )
#path.append( (105.07658386230469, 129.310546875, 0.0) )
path.append( (105.07658386230469, 129.7857208251953, 0.0) )
#path.append( (104.12622833251953, 129.7857208251953, 0.0) )
#path.append( (100.80000305175781, 129.7857208251953, 0.0) )
#path.append( (95.57306671142578, 129.7857208251953, 0.0) )
path.append( (93.19718933105469, 129.7857208251953, 0.0) )
#path.append( (92.72201538085938, 129.7857208251953, 0.0) )
#path.append( (92.24684143066406, 129.7857208251953, 0.0) )
#path.append( (91.77165985107422, 129.7857208251953, 0.0) )
path.append( (89.87095642089844, 129.310546875, 0.0) )
#path.append( (88.92060852050781, 129.310546875, 0.0) )
#path.append( (88.4454345703125, 129.310546875, 0.0) )
#path.append( (86.54473114013672, 129.310546875, 0.0) )
path.append( (86.06954956054688, 129.310546875, 0.0) )
#path.append( (85.59437561035156, 129.310546875, 0.0) )
#path.append( (85.11920166015625, 129.310546875, 0.0) )
#path.append( (85.59437561035156, 129.7857208251953, 0.0) )

pathpoints = vtk.vtkPoints()
for p in path:
    pathpoints.InsertNextPoint(p[0],p[1],sliceno * v16.GetOutput().GetSpacing()[2])

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
# print ("mask0", mask0.GetOutput().GetScalarTypeAsString())
# print ("mask1", mask1.GetOutput().GetScalarTypeAsString())
# print ("image", v16.GetOutput().GetScalarTypeAsString())

lasso = vtk.vtkLassoStencilSource()
lasso.SetShapeToPolygon()
# pass the slice at which the lasso has to process
lasso.SetSlicePoints(sliceno , pathpoints)
lasso.SetInformationInput(v16.GetOutput())

stencil = vtk.vtkImageStencil()
stencil.SetInputConnection(mask1.GetOutputPort())
stencil.SetBackgroundInputData(mask0.GetOutput())
stencil.SetStencilConnection(lasso.GetOutputPort())
stencil.Update()






points = createPoints(density, sliceno , v16.GetOutput(), orientation)

print ("createdPoints ", points.GetPoint(1))
print ("spacing" , v16.GetOutput().GetSpacing())

#%%




vertices = points2vertices(points)


pointPolyData = vtk.vtkPolyData()
pointPolyData.SetPoints(points)
pointPolyData.SetVerts(vertices)

pointCloud = cilRegularPointCloudToPolyData()
pointCloud.SetMode(cilRegularPointCloudToPolyData.CUBE)
pointCloud.SetDimensionality(2)
pointCloud.SetSlice(3)
pointCloud.SetInputConnection(0, v16.GetOutputPort())
pointCloud.SetOverlap(0,0.3)
pointCloud.SetOverlap(1,0.5)
pointCloud.SetOverlap(2,0.4)
pointCloud.SetSubVolumeRadiusInVoxel(3)
pointCloud.Update()

print ("pointCloud number of points", pointCloud.GetNumberOfPoints())
     

rotate = (0,0.,25)
transform = vtk.vtkTransform()
transform.Translate(dimensions[0]/2*spacing[0], dimensions[1]/2*spacing[1],0)
transform.RotateX(rotate[0])
transform.RotateY(rotate[1])
transform.RotateZ(rotate[2])
transform.Translate(-dimensions[0]/2*spacing[0], -dimensions[1]/2*spacing[1],0)

t_filter = vtk.vtkTransformFilter()
t_filter.SetTransform(transform)
#t_filter.SetInputData(pointPolyData)
t_filter.SetInputConnection(pointCloud.GetOutputPort())
#t_filter.Update()





# masked_polydata = vtkMaskPolyData()
# masked_polydata.SetMask(stencil.GetOutput())
# masked_polydata.SetPolyDataInput(t_filter.GetOutput())

print (stencil.GetOutput())
# Erode the mask of SubVolumeRadius
erode = vtk.vtkImageDilateErode3D()
erode.SetInputConnection(0,stencil.GetOutputPort())
erode.SetErodeValue(1)
erode.SetDilateValue(0) #: shouldn't exist in the mask
ks = [pointCloud.GetSubVolumeRadiusInVoxel(), pointCloud.GetSubVolumeRadiusInVoxel(), 0]
if pointCloud.GetDimensionality() == 3:
    ks[2]= pointCloud.GetSubVolumeRadiusInVoxel()

print (ks)
erode.SetKernelSize(ks[0],ks[1],1)
erode.Update()
print (erode.GetOutput())

polydata_masker = cilMaskPolyData()
polydata_masker.SetMaskValue(1)
polydata_masker.SetInputConnection(0, t_filter.GetOutputPort())
polydata_masker.SetInputConnection(1, erode.GetOutputPort())
# polydata_masker.SetInputConnection(1, stencil.GetOutputPort())
polydata_masker.Update()

mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputConnection(t_filter.GetOutputPort())
# mapper.SetInputData(masked_polydata.GetOutput())
mapper.SetInputConnection(polydata_masker.GetOutputPort())
print ("masked point", polydata_masker.point_in_mask)

actor = vtk.vtkLODActor()
actor.SetMapper(mapper)
actor.GetProperty().SetPointSize(3)
actor.GetProperty().SetColor(1, .2, .2)
actor.VisibilityOn()


### Sphere glyphs for pointcloud
# subvolume
# arrow
subv_glyph = vtk.vtkGlyph3D()
subv_glyph.SetScaleFactor(1.)
# arrow_glyph.SetColorModeToColorByVector()
spacing = erode.GetOutput().GetSpacing()
radius = pointCloud.GetSubVolumeRadiusInVoxel()

sphere_source = vtk.vtkSphereSource()
sphere_source.SetRadius(radius * spacing[0])
sphere_source.SetThetaResolution(12)
sphere_source.SetPhiResolution(12)

print ("spacing",spacing)
print ("radius",radius)

cube_source = vtk.vtkCubeSource()
cube_source.SetXLength(spacing[0]*radius)
cube_source.SetYLength(spacing[1]*radius)
cube_source.SetZLength(spacing[2]*radius)

sphere_mapper = vtk.vtkPolyDataMapper()
sphere_mapper.SetInputConnection( subv_glyph.GetOutputPort() )
subv_glyph.SetInputConnection(polydata_masker.GetOutputPort())
# subv_glyph.SetSourceConnection( sphere_source.GetOutputPort() )
subv_glyph.SetSourceConnection( cube_source.GetOutputPort() )

# Usual actor
sphere_actor = vtk.vtkActor()
sphere_actor.SetMapper(sphere_mapper)
sphere_actor.GetProperty().SetColor(1, 0, 0)
sphere_actor.GetProperty().SetOpacity(0.2)


if True:
    v = CILViewer2D()
    # v.setInput3DData(v16.GetOutput())
    v.setInput3DData(stencil.GetOutput())
    # v.setInput3DData(erode.GetOutput())
    v.ren.AddActor(actor)
    v.ren.AddActor(sphere_actor)
    v.startRenderLoop()