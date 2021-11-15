# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:57:35 2020

@author: chenlin
"""

from matplotlib import pyplot as plt
from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon
import numpy as np

def create_ellipse(center, lengths, angle=0):
    """
    create a shapely ellipse. adapted from
    https://gis.stackexchange.com/a/243462
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr

fig,ax = plt.subplots()

##these next few lines are pretty important because
##otherwise your ellipses might only be displayed partly
##or may be distorted
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_aspect('equal')

##first ellipse in blue
ellipse1 = create_ellipse((0,0),(2,4),10)
verts1 = np.array(ellipse1.exterior.coords.xy)
patch1 = Polygon(verts1.T, color = 'blue', alpha = 0.5)
ax.add_patch(patch1)

##second ellipse in red    
ellipse2 = create_ellipse((1,-1),(3,2),50)
verts2 = np.array(ellipse2.exterior.coords.xy)
patch2 = Polygon(verts2.T,color = 'red', alpha = 0.5)
ax.add_patch(patch2)

##the intersect will be outlined in black
intersect = ellipse1.intersection(ellipse2)
verts3 = np.array(intersect.exterior.coords.xy)
patch3 = Polygon(verts3.T, facecolor = 'none', edgecolor = 'black')
ax.add_patch(patch3)

##compute areas and ratios 
print('area of ellipse 1:',ellipse1.area)
print('area of ellipse 2:',ellipse2.area)
print('area of intersect:',intersect.area)
print('intersect/ellipse1:', intersect.area/ellipse1.area)
print('intersect/ellipse2:', intersect.area/ellipse2.area)


plt.show()