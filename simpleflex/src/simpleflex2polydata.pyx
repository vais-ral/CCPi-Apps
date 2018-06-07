# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 06:43:46 2018

@author: ofn77899
"""

import cython
import vtk
import numpy
cimport numpy

def s2pd(origin, spacing, surf_list):
    #Display isosurfaces in 3D
    # Create the VTK output
    # Points coordinates structure
    triangle_vertices = vtk.vtkPoints()

    # associate the points to triangles
    triangle = vtk.vtkTriangle()
    trianglePointId = triangle.GetPointIds()
    
    # put all triangles in an array
    triangles = vtk.vtkCellArray()
    cdef int isTriangle = 0
    cdef int nTriangle = 0

    cdef int surface = 0
    # associate each coordinate with a point: 3 coordinates are needed for a point
    # in 3D. Additionally we perform a shift from image coordinates (pixel) which
    # is the default of the Contour Tree Algorithm to the World Coordinates.

    
    # augmented matrix for affine transformations
    mScaling = numpy.asarray([spacing[0], 0, 0, 0,
                              0, spacing[1], 0, 0,
                              0, 0, spacing[2], 0,
                              0, 0, 0, 1]).reshape((4, 4))
    mShift = numpy.asarray([1, 0, 0, origin[0],
                            0, 1, 0, origin[1],
                            0, 0, 1, origin[2],
                            0, 0, 0, 1]).reshape((4, 4))

    cdef numpy.ndarray mTransform = numpy.dot(mScaling, mShift)
    cdef int point_count = 0
    cdef int i = 0
    cdef numpy.ndarray point 
    cdef numpy.ndarray world_coord
    
    
    for surf in surf_list:
        print("Image-to-world coordinate trasformation ... %d" % surface)
        for i in range(len(surf)):
            point = surf[i]
            world_coord = numpy.dot(mTransform, point)
            #xCoord = world_coord[0]
            #yCoord = world_coord[1]
            #zCoord = world_coord[2]
            #xCoord = point[0] * (xSpacing) + xOrigin;
            #yCoord = point[1] * (ySpacing) + yOrigin;
            #zCoord = point[2] * (zSpacing) + zOrigin;
            triangle_vertices.InsertNextPoint(world_coord[0], 
                                              world_coord[1], 
                                              world_coord[2]);

            # The id of the vertex of the triangle (0,1,2) is linked to
            # the id of the points in the list, so in facts we just link id-to-id
            trianglePointId.SetId(isTriangle, point_count)
            isTriangle += 1
            point_count += 1

            if (isTriangle == 3):
                isTriangle = 0;
                # insert the current triangle in the triangles array
                triangles.InsertNextCell(triangle);

        surface += 1

    # polydata object
    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(triangle_vertices)
    trianglePolyData.SetPolys(triangles)
    return trianglePolyData