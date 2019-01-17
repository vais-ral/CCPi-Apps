# -*- coding: utf-8 -*-
"""
CILViewer utils

Created on Wed Jan 16 14:21:00 2019

@author: Edoardo Pasca
"""
import os
from numbers import Integral, Number
import numpy
import vtk
from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase


class cilRegularPointCloudToPolyData(VTKPythonAlgorithmBase):
    '''vtkAlgorithm to create a regular point cloud grid for Digital Volume Correlation

    In DVC points between a reference volume and a correlation volume are correlated.
    The DVC process requires to track points whithin a subvolume of the entire
    volume that are around each point. For instance, points within a sphere of
    a certain radius (in voxel) around a point are part of the subvolume.

    The regular point cloud grid is laid out based on the overlap between
    two consecutive subvolumes. The overlap can be set indipendently on each
    axis.
    The user can provide the shape of the subvolume and the radius (for cubes
    the radius is the length of the side).

    Example:
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

    '''
    CIRCLE = 'circle'
    SQUARE = 'square'
    CUBE   = 'cube'
    SPHERE = 'sphere'
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1)
        self.__Points = vtk.vtkPoints()
        self.__Vertices = vtk.vtkCellArray()
        self.__Orientation = 2
        self.__Overlap = [0.2, 0.2, 0.2] #: 3D overlap
        self.__Dimensionality = 3
        self.__SliceNumber = 0
        self.__Mode = self.CUBE
        self.__SubVolumeRadius = 1 #: Radius of the subvolume in voxels

    def GetPoints(self):
        '''Returns the Points'''
        return self.__Points
    def SetMode(self, value):
        '''Sets the shape mode'''
        if not value in [self.CIRCLE, self.SQUARE, self.CUBE, self.SPHERE]:
            raise ValueError('dimension must be in [circle, square, cube, sphere]. Got',
                             value)

        if value != self.__Mode:
            self.__Mode = value
            self.Modified()

    def GetMode(self):
        return self.__Mode

    def SetDimensionality(self, value):
        '''Whether the overlap is measured on 2D or 3D'''
        if not value in [2, 3]:
            raise ValueError('Dimensionality must be in [2, 3]. Got', value)
        if self.__Dimensionality != value:
            self.__Dimensionality = value
            self.Modified()
    def GetDimensionality(self):
        return self.__Dimensionality

    def SetOverlap(self, dimension, value):
        '''Set the overlap between'''
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
        '''For 2D represents the slice on the data where you want to get points laid out'''
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
        '''returns the number of points in the point cloud'''
        return self.__Points.GetNumberOfPoints()

    def SetOrientation(self, value):
        '''For 2D sets the orientation of the working plane'''
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

        # print ("Request Data")
        image_data = vtk.vtkDataSet.GetData(inInfo[0])
        pointPolyData = vtk.vtkPolyData.GetData(outInfo)

        # print ("orientation", orientation)
        dimensionality = self.GetDimensionality()
        # print ("dimensionality", dimensionality)

        overlap = self.GetOverlap()
        # print ("overlap", overlap)
        point_spacing = self.CalculatePointSpacing(overlap, mode=self.GetMode())
        # print ("point_spacing", point_spacing)

        if dimensionality == 3:
            self.CreatePoints3D(point_spacing, image_data)
        else:
            sliceno = self.GetSlice()
            orientation = self.GetOrientation()

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
            point_spacing: distance between points in voxels (list or tuple)
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
        # dim_z     = image_dimensions.pop(orientation)

        # the total number of points on X and Y axis
        max_x = int(image_dimensions[0] / point_spacing[0] )
        max_y = int(image_dimensions[1] / point_spacing[1] )

        z = sliceno * spacing_z - origin_z

        # skip the offset in voxels
        offset = [0, 0]
        n_x = offset[0]

        while n_x < max_x:
            # x axis
            n_y = offset[1]
            while n_y < max_y:
                # y axis
                x = (n_x / max_x) * image_spacing[0] * image_dimensions[0]- image_origin[0] #+ int(image_dimensions[0] * density[0] * .7)
                y = (n_y / max_y) * image_spacing[1] * image_dimensions[1]- image_origin[1] #+ int(image_dimensions[1] * density[1] * .7)

                vtkPointCloud.InsertNextPoint( x , y , z)


                n_y += 1

            n_x += 1

        return 1

    def CreatePoints3D(self, point_spacing , image_data ):
        '''creates a 3D point cloud on the image data on the selected orientation

        input:
            point_spacing: distance between points in voxels (list or tuple)
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

        # print ("max_x: {} {} {}".format((max_x, max_y, max_z), image_dimensions, point_spacing))
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

                    vtkPointCloud.InsertNextPoint( x, y, z )
                    n_z += 1

                n_y += 1

            n_x += 1

        return 1
    def FillCells(self):
        '''Fills the Vertices'''
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
        '''Calculates the volume overlap for 2 shapes of radius and center distance'''
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
        '''hard inversion of distance and overlap'''
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


class cilMaskPolyData(VTKPythonAlgorithmBase):
    '''vtkAlgorithm to crop a vtkPolyData with a Mask

    This is really only meant for point clouds: see points2vertices function
    '''
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=2, nOutputPorts=1)
        self.__MaskValue = 1

    def SetMaskValue(self, mask_value):
        '''Sets the value at which the mask is active'''
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

                if int(mm) == int(self.GetMaskValue()):
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
        self.__PlaneOriginAbove    = 0
        self.__PlaneNormalAbove    = 1
        self.__PlaneOriginBelow    = 0
        self.__PlaneNormalBelow    = 1
        
        self.planesource = [ vtk.vtkPlaneSource(), vtk.vtkPlaneSource() ]
        self.visPlane = [ vtk.vtkPlane() , vtk.vtkPlane() ]
        self.planeClipper =  [ vtk.vtkClipPolyData() , vtk.vtkClipPolyData()]

        self.planesource[0].SetCenter(self.visPlane[0].GetOrigin())
        self.planesource[1].SetCenter(self.visPlane[1].GetOrigin())
        self.planesource[0].SetNormal(self.visPlane[0].GetNormal())
        self.planesource[1].SetNormal(self.visPlane[1].GetNormal())
        
        self.planeClipper[0].SetClipFunction(self.visPlane[0])
        self.planeClipper[1].SetClipFunction(self.visPlane[1])
        
        self.planeClipper[0].InsideOutOn()
        self.planeClipper[1].InsideOutOn()
            

    def SetPlaneOriginAbove(self, value):
        if not (isinstance(value, list) or isinstance(value, tuple)):
            raise ValueError('Spacing should be a list or a tuple. Got', type(value))

        if value != self.__PlaneOriginAbove:
            self.__PlaneOriginAbove = value
            self.Modified()

    def GetPlaneOriginAbove(self):
        return self.__PlaneOriginAbove
    def SetPlaneNormalAbove(self, value):
        if not (isinstance(value, list) or isinstance(value, tuple)):
            raise ValueError('Spacing should be a list or a tuple. Got', type(value))

        if value != self.__PlaneNormalAbove:
            self.__PlaneNormalAbove = value
            self.Modified()
    def GetPlaneNormalAbove(self):
        return self.__PlaneNormalAbove
    def SetPlaneOriginBelow(self, value):
        if not (isinstance(value, list) or isinstance(value, tuple)):
            raise ValueError('Spacing should be a list or a tuple. Got', type(value))

        if value != self.__PlaneOriginBelow:
            self.__PlaneOriginBelow = value
            self.Modified()
    def GetPlaneOriginBelow(self):
        return self.__PlaneOriginBelow
    def SetPlaneNormalBelow(self, value):
        if not (isinstance(value, list) or isinstance(value, tuple)):
            raise ValueError('Spacing should be a list or a tuple. Got', type(value))

        if value != self.__PlaneNormalBelow:
            self.__PlaneNormalBelow = value
            self.Modified()
    def GetPlaneNormalBelow(self):
        return self.__PlaneNormalBelow

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
        self.planeClipper[0].SetInputData(vtk.vtkPolyData.GetData(inInfo[0]))
        self.planeClipper[1].SetInputConnection(self.planeClipper[0].GetOutputPort())
        
        origin = self.GetPlaneOriginAbove()
        normal = self.GetPlaneNormalAbove()
        self.visPlane[0].SetOrigin(origin[0], origin[1], origin[2])
        self.visPlane[0].SetNormal(normal[0], normal[1], normal[2])
        
        origin = self.GetPlaneOriginBelow()
        normal = self.GetPlaneNormalBelow()
        self.visPlane[1].SetOrigin(origin[0], origin[1], origin[2])
        self.visPlane[1].SetNormal(normal[0], normal[1], normal[2])
        
        pointPolyData = vtk.vtkPolyData.GetData(outInfo)
        pointPolyData.DeepCopy(self.planeClipper[1].GetOutput())
        return 1


class cilNumpyMETAImageWriter():
    '''A Writer to write a Numpy Array in npy format and a METAImage Header

    This way the same data file can be accessed by NumPy and VTK
    '''
    __FileName = None
    __Array = None
    __Spacing = (1.,1.,1.)
    def __init__(self):
        pass

    def SetInputData(self, array):
        if not isinstance (array, numpy.ndarray):
            raise ValueError('Input must be a NumPy array. Got' , type(array))
        self.__Array = array
    def Write(self):
        if self.__Array is None:
            raise ValueError('Array is None')
        if self.__FileName is None:
            raise ValueError('FileName is None')

        WriteNumpyAsMETAImage(self.__Array, self.__FileName, self.__Spacing)
    def SetFileName(self, fname):
        self.__FileName = os.path.abspath(fname)
    def GetFileName(self):
        return self.__FileName
    def SetSpacing(self, value):
        if not (isinstance(value, list) or isinstance(value, tuple)):
            raise ValueError('Spacing should be a list or a tuple. Got', type(value))
        if len(value) != len(self.__Array.shape):
            self.__Spacing = value
            self.Modified()
        

def WriteNumpyAsMETAImage(array, filename, spacing=(1.,1.,1.)):
    '''Writes a NumPy array and a METAImage text header so that the npy file can be used as data file'''
    # save the data as numpy
    datafname = os.path.abspath(filename) + '.npy'
    hdrfname =  os.path.abspath(filename) + '.mhd'
    if (numpy.isfortran(array)):
        numpy.save(datafname, array)
    else:
        numpy.save(datafname, numpy.asfortranarray(array))
    npyhdr = parseNpyHeader(datafname)

    # inspired by
    # https://github.com/vejmarie/vtk-7/blob/master/Wrapping/Python/vtk/util/vtkImageImportFromArray.py
    # and metaTypes.h in VTK source code
    __typeDict = {'b':'MET_CHAR',    # VTK_SIGNED_CHAR,     # int8
                  'B':'MET_UCHAR',   # VTK_UNSIGNED_CHAR,   # uint8
                  'h':'MET_SHORT',   # VTK_SHORT,           # int16
                  'H':'MET_USHORT',  # VTK_UNSIGNED_SHORT,  # uint16
                  'i':'MET_INT',     # VTK_INT,             # int32
                  'I':'MET_UINT',    # VTK_UNSIGNED_INT,    # uint32
                  'f':'MET_FLOAT',   # VTK_FLOAT,           # float32
                  'd':'MET_DOUBLE',  # VTK_DOUBLE,          # float64
                  'F':'MET_FLOAT',   # VTK_FLOAT,           # float32
                  'D':'MET_DOUBLE'   # VTK_DOUBLE,          # float64
          }

    typecode = array.dtype.char
    print ("typecode,",typecode)
    ar_type = __typeDict[typecode]
    # save header
    # minimal header structure
    # NDims = 3
    # DimSize = 181 217 181
    # ElementType = MET_UCHAR
    # ElementSpacing = 1.0 1.0 1.0
    # ElementByteOrderMSB = False
    # ElementDataFile = brainweb1.raw
    header = 'ObjectType = Image\n'
    header = ''
    header += 'NDims = {0}\n'.format(len(array.shape))
    header += 'DimSize = {} {} {}\n'.format(array.shape[0], array.shape[1], array.shape[2])
    header += 'ElementType = {}\n'.format(ar_type)
    header += 'ElementSpacing = {} {} {}\n'.format(spacing[0], spacing[1], spacing[2])
    # MSB (aka big-endian)
    descr = npyhdr['description']
    MSB = 'True' if descr['descr'][0] == '>' else 'False'
    header += 'ElementByteOrderMSB = {}\n'.format(MSB)

    header += 'HeaderSize = {}\n'.format(npyhdr['header_length'])
    header += 'ElementDataFile = {}'.format(os.path.basename(datafname))

    with open(hdrfname , 'w') as hdr:
        hdr.write(header)

#%%


def parseNpyHeader(filename):
    '''parses a npy file and returns a dictionary with version, header length and description

    See https://www.numpy.org/devdocs/reference/generated/numpy.lib.format.html for details
    of information included in the output.
    '''
    import struct
    with open(filename, 'rb') as f:
        c = f.read(6)
        if not c == b"\x93NUMPY":
            raise TypeError('File Type is not npy')
        major = struct.unpack('@b', f.read(1))[0]
        minor = struct.unpack('@b', f.read(1))[0]
        if major == 1:
            HEADER_LEN_SIZE = 2
        elif major == 2:
            HEADER_LEN_SIZE = 4

        # print ('NumPy file version {}.{}'.format(major, minor))
        HEADER_LEN = struct.unpack('<H', f.read(HEADER_LEN_SIZE))[0]
        # print ("header_len", HEADER_LEN, type(HEADER_LEN))
        descr = ''
        i = 0
    with open(filename, 'r') as f:
        f.seek(6+2 + HEADER_LEN_SIZE)

        while i < HEADER_LEN:
            c = f.read(1)
            #print (c)
            descr += c
            i += 1
    return {'type': 'NUMPY',
            'version_major':major,
            'version_minor':minor,
            'header_length':HEADER_LEN + 6 + 2 + HEADER_LEN_SIZE,
            'description'  : eval(descr)}


if __name__ == '__main__':

    import os
    dimX = 128
    dimY = 64
    dimZ = 32
    # a = numpy.zeros((dimX,dimY,dimZ), dtype=numpy.uint16)
    a = numpy.random.randint(0,size=(dimX,dimY,dimZ),high=127,  dtype=numpy.uint16)
    #for x in range(a.shape[0]):
    #    for y in range(a.shape[1]):
    #        for z in range(a.shape[2]):
    #            a[x][y][z] = x + a.shape[0] * y + a.shape[0]*a.shape[1]*z
    for z in range(a.shape[2]):
        b = z * numpy.ones((a.shape[0],a.shape[1]), dtype=numpy.uint8)
        a[:,:,z] = b.copy()

    #arfn = os.path.abspath('C:/Users/ofn77899/Documents/Projects/CCPi/GitHub/CCPi-Simpleflex/data/head.npy')
    arfn = 'test'
    WriteNumpyAsMETAImage(a,arfn)
    hdrdescr = parseNpyHeader(arfn+'.npy')

    #a = numpy.load(arfn+'.npy')

    #import matplotlib.pyplot as plt
    #plt.imshow(a[10,:,:])
    #plt.show()

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(arfn+'.mhd')
    reader.Update()



    if True:
        from ccpi.viewer.CILViewer2D import CILViewer2D, Converter
        v = CILViewer2D()
        v.setInput3DData(reader.GetOutput())
        v.startRenderLoop()
    if False:
        #x = 120
        #y = 60
        #z = 30
        #v1 = a[x][y][z]
        #v2 = numpy.uint8(reader.GetOutput().GetScalarComponentAsFloat(x,y,z,0))
        #print ("value check", v1,v2)
        is_same = a.shape == reader.GetOutput().GetDimensions()

        for x in range(a.shape[0]):
            for y in range(a.shape[1]):
                for z in range(a.shape[2]):
                    v1 = a[x][y][z]
                    v2 = numpy.uint8(reader.GetOutput().GetScalarComponentAsFloat(x,y,z,0))
                    # print ("value check {} {} {},".format((x,y,z) ,v1,v2))
                    is_same = v1 == v2
                    if not is_same:
                        raise ValueError('arrays do not match', v1,v2,x,y,z)
        print ('YEEE array match!')

