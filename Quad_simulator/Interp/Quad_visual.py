#!/usr/bin/env python
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import sys
import numpy as np
from math import pi as PI
from math import atan2, sin, cos, sqrt, fabs, floor, exp
from scipy.misc import factorial as ft
from scipy.misc import comb as nCk
from numpy import linalg as LA
from numpy.linalg import inv

import matplotlib.patches
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to 
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector    
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])
    #print pathpatch._segment3d

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta

class Shpere3D():
    def __init__(self, initpos, ax):
        u, v = np.mgrid[0:2*PI:9j, 0:PI:5j]
        self.zscale = 3
        self.x = 0.3/2*np.sign(np.cos(u)*np.sin(v))*np.sqrt(np.abs(np.cos(u)*np.sin(v))) 
        self.y = 0.3/2*np.sign(np.sin(u)*np.sin(v))*np.sqrt(np.abs(np.sin(u)*np.sin(v)))
        self.z = 0.3/2*self.zscale*np.sign(np.cos(v))*np.sqrt(np.abs(np.cos(v)))
        self.handle = ax.plot_wireframe(self.x+initpos[0], self.y+initpos[1], self.z+initpos[2], color="k")
        #plt.pause(.001)  #moved outside of the for loop

    def update(self, newpos, ax):
        oldcol = self.handle
        ax.collections.remove(oldcol)
        self.handle = ax.plot_wireframe(self.x+newpos[0], self.y+newpos[1], self.z+newpos[2], color="k")
        #plt.pause(.001)  #moved outside of the for loop
        #plt.draw()

class Coord3D():
    def __init__(self, ax, initpos, roll=0.0, pitch=0.0, yaw=0.0):
        self.r0 = 0.15#0.07
        self.r1 = 0.08#0.08/2
        R = self.RotMat()
        L = 0.35
        self.hx, = plt.plot([initpos[0] , initpos[0]+L*R[0,0]], [initpos[1] , initpos[1]+L*R[1,0]], [initpos[2] , initpos[2]+L*R[2,0]], 'r')
        self.hy, = plt.plot([initpos[0] , initpos[0]+L*R[0,1]], [initpos[1] , initpos[1]+L*R[1,1]], [initpos[2] , initpos[2]+L*R[2,1]], 'g')
        self.hz, = plt.plot([initpos[0] , initpos[0]+L*R[0,2]], [initpos[1] , initpos[1]+L*R[1,2]], [initpos[2] , initpos[2]+L*R[2,2]], 'b')
        d = np.cross((R[0,2], R[1,2], R[2,2]),(0, 0, 1)) #Obtain the rotation vector    
        self.M = rotation_matrix(d) #Get the rotation matrix
        self.pos = initpos
        self.disk1 = matplotlib.patches.Circle((initpos[0]+self.r0/sqrt(2), initpos[1]+self.r0/sqrt(2)), self.r1)
        self.disk2 = matplotlib.patches.Circle((initpos[0]+self.r0/sqrt(2), initpos[1]-self.r0/sqrt(2)), self.r1)
        self.disk3 = matplotlib.patches.Circle((initpos[0]-self.r0/sqrt(2), initpos[1]+self.r0/sqrt(2)), self.r1)
        self.disk4 = matplotlib.patches.Circle((initpos[0]-self.r0/sqrt(2), initpos[1]-self.r0/sqrt(2)), self.r1)
        ax.add_patch(self.disk1)
        ax.add_patch(self.disk2)
        ax.add_patch(self.disk3)
        ax.add_patch(self.disk4)
        pathpatch_2d_to_3d(self.disk1, z=initpos[2], normal=((R[0,2],R[1,2],R[2,2])))
        pathpatch_2d_to_3d(self.disk2, z=initpos[2], normal=((R[0,2],R[1,2],R[2,2])))
        pathpatch_2d_to_3d(self.disk3, z=initpos[2], normal=((R[0,2],R[1,2],R[2,2])))
        pathpatch_2d_to_3d(self.disk4, z=initpos[2], normal=((R[0,2],R[1,2],R[2,2])))

    def RotMat(self, r=0.0,p=0.0,y=0.0):
        R = np.array([[cos(p)*cos(y), sin(r)*sin(p)*cos(y)-cos(r)*sin(y), cos(r)*sin(p)*cos(y)+sin(r)*sin(y)],
                      [cos(p)*sin(y), sin(r)*sin(p)*sin(y)+cos(r)*cos(y), cos(r)*sin(p)*sin(y)-sin(r)*cos(y)],
                      [-sin(p), sin(r)*cos(p), cos(r)*cos(p)]])
        return R

    def update(self, pos, r, p, y=0):
        R = self.RotMat(r, p, y)
        L = 0.35
        self.hx.set_xdata([pos[0] , pos[0]+L*R[0,0]])
        self.hx.set_ydata([pos[1] , pos[1]+L*R[1,0]])
        self.hx.set_3d_properties([pos[2] , pos[2]+L*R[2,0]])
        self.hy.set_xdata([pos[0] , pos[0]+L*R[0,1]])
        self.hy.set_ydata([pos[1] , pos[1]+L*R[1,1]])
        self.hy.set_3d_properties([pos[2] , pos[2]+L*R[2,1]])
        self.hz.set_xdata([pos[0] , pos[0]+L*R[0,2]])
        self.hz.set_ydata([pos[1] , pos[1]+L*R[1,2]])
        self.hz.set_3d_properties([pos[2] , pos[2]+L*R[2,2]])
        d = np.cross((R[0,2], R[1,2], R[2,2]),(0, 0, 1)) #Obtain the rotation vector    
        newM = rotation_matrix(d) #Get the rotation matrix
        self.disk1._segment3d -= np.kron(np.ones((self.disk1._segment3d[:,0].size,1)), self.pos)
        self.disk2._segment3d -= np.kron(np.ones((self.disk2._segment3d[:,0].size,1)), self.pos)
        self.disk3._segment3d -= np.kron(np.ones((self.disk3._segment3d[:,0].size,1)), self.pos)
        self.disk4._segment3d -= np.kron(np.ones((self.disk4._segment3d[:,0].size,1)), self.pos)
        self.disk1._segment3d = np.dot(newM, np.dot(inv(self.M), self.disk1._segment3d.T)).T
        self.disk2._segment3d = np.dot(newM, np.dot(inv(self.M), self.disk2._segment3d.T)).T
        self.disk3._segment3d = np.dot(newM, np.dot(inv(self.M), self.disk3._segment3d.T)).T
        self.disk4._segment3d = np.dot(newM, np.dot(inv(self.M), self.disk4._segment3d.T)).T
        self.disk1._segment3d += np.kron(np.ones((self.disk1._segment3d[:,0].size,1)), pos)
        self.disk2._segment3d += np.kron(np.ones((self.disk2._segment3d[:,0].size,1)), pos)
        self.disk3._segment3d += np.kron(np.ones((self.disk3._segment3d[:,0].size,1)), pos)
        self.disk4._segment3d += np.kron(np.ones((self.disk4._segment3d[:,0].size,1)), pos)
        #np.array([np.dot(newM, np.dot(inv(self.M), (x, y, 0))) + (0, 0, pos[2]) for x, y in verts])
        self.pos = pos
        self.M = newM

class Rect2D():
    def __init__(self, ax, pos, tail, c = 'g'):
        theta = np.arange(0,2*PI+PI/20,PI/20)
        self.r0 = 0.07
        self.r1 = 0.08/2
        self.x = 0.2/2*np.sign(np.cos(theta))*np.sqrt(np.abs(np.cos(theta))) 
        self.y = 0.2/2*np.sign(np.sin(theta))*np.sqrt(np.abs(np.sin(theta)))
        self.h1, = ax.plot([pos[0]+self.r0/sqrt(2), pos[0]-self.r0/sqrt(2)], [pos[1]+self.r0/sqrt(2), pos[1]-self.r0/sqrt(2)], c, linewidth=1)
        self.h2, = ax.plot([pos[0]+self.r0/sqrt(2), pos[0]-self.r0/sqrt(2)], [pos[1]-self.r0/sqrt(2), pos[1]+self.r0/sqrt(2)], c, linewidth=1)
        self.ht, = ax.plot(tail[:,0], tail[:,1], c, linewidth=2)
        self.handle, = ax.plot(self.x+pos[0], self.y+pos[1], c, linestyle = 'solid', linewidth=0.5)
        self.disk1 = matplotlib.patches.Circle(
            (pos[0]+self.r0/sqrt(2), pos[1]+self.r0/sqrt(2)),
            radius = self.r1,
            edgecolor= c,#'#afeeee',
            fill = False,
            lw = 1,
            alpha=0.6)
        ax.add_patch(self.disk1)
        self.disk2 = matplotlib.patches.Circle(
            (pos[0]+self.r0/sqrt(2), pos[1]-self.r0/sqrt(2)),
            radius = self.r1,
            edgecolor= c,#'#afeeee',
            fill = False,
            lw = 1,
            alpha=0.6)
        ax.add_patch(self.disk2)
        self.disk3 = matplotlib.patches.Circle(
            (pos[0]-self.r0/sqrt(2), pos[1]+self.r0/sqrt(2)),
            radius = self.r1,
            edgecolor= c,#'#afeeee',
            fill = False,
            lw = 1,
            alpha=0.6)
        ax.add_patch(self.disk3)
        self.disk4 = matplotlib.patches.Circle(
            (pos[0]-self.r0/sqrt(2), pos[1]-self.r0/sqrt(2)),
            radius = self.r1,
            edgecolor= c,#'#afeeee',
            fill = False,
            lw = 1,
            alpha=0.6)
        ax.add_patch(self.disk4)

    def update(self, pos, tail):
        self.handle.set_data(self.x+pos[0], self.y+pos[1])
        self.disk1.center = pos[0]+self.r0/sqrt(2), pos[1]+self.r0/sqrt(2)
        self.disk2.center = pos[0]+self.r0/sqrt(2), pos[1]-self.r0/sqrt(2)
        self.disk3.center = pos[0]-self.r0/sqrt(2), pos[1]+self.r0/sqrt(2)
        self.disk4.center = pos[0]-self.r0/sqrt(2), pos[1]-self.r0/sqrt(2)
        self.h1.set_data([pos[0]+self.r0/sqrt(2), pos[0]-self.r0/sqrt(2)], [pos[1]+self.r0/sqrt(2), pos[1]-self.r0/sqrt(2)])
        self.h2.set_data([pos[0]+self.r0/sqrt(2), pos[0]-self.r0/sqrt(2)], [pos[1]-self.r0/sqrt(2), pos[1]+self.r0/sqrt(2)])
        self.ht.set_data(tail[:,0], tail[:,1])


if __name__ == "__main__":
    fig = plt.figure()
    ax  = fig.gca(projection = '3d')
    ax.set_aspect('equal')
    ax.invert_zaxis()
    N = 1 # consider 1 quad as an example

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([3, 2, 1.6]).max()
    Xb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][0].flatten()
    Yb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][1].flatten() 
    Zb = 0.5*max_range*np.mgrid[-1:1:2j,-1:1:2j,-1:1:2j][2].flatten() - 1
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')


    # Draw a circle on the x=0 'wall'
    #p = matplotlib.patches.Circle((0, 0), 0.5)
    #ax.add_patch(p)
    #pathpatch_2d_to_3d(p, z=0, normal=((1,1,1)))
    #art3d.pathpatch_translate(p, (0.5, 1, 0))

    plt.draw()

    # initial point
    p0 = dict()
    p0[0] = np.array([[-1, 0, -0.5],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])


    # init ploting handles
    Cfplt = dict() 
    CfCoord = dict()
    for i in range(N):
        Cfplt[i] = Shpere3D(p0[i][0,:], ax)
        CfCoord[i] = Coord3D(ax, p0[i][0,:])




    plt.show()






