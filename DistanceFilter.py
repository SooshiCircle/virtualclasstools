# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 21:45:29 2020

@author: alexa
"""
import math

#Trivial Stuff

ERROR_THRESHOLD = 0

#Distance between two coordinates 
def dist(l1, l2):
    return math.sqrt((l1[0]-l2[0])**2+(l1[1]-l2[1])**2)

#Checks whether new point is close enough to previous
def filter(listy3, coordToAdd): 
    return dist(listy3[-1], coordToAdd) < ERROR_THRESHOLD


listy = [[1,5], [3,4], [8,8]] #List of coordinates 



        
        
        
print("Dist: "+str(dist([1,1],[4,5])))