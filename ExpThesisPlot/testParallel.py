import numpy as np
from multiprocessing import Pool # multiprocessing
from itertools import product


class Test():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
     
    def update(self, newx, newy):
        self.x = newx
        self.y = newy
   
# used to test parallel computing
def mod(obj, newx, newy):
    obj.x = newx
    obj.y = newy
    #print obj.x

# unpack args to fix bugs
def mod_unpack(args):
    mod(*args)


if __name__ == "__main__":
    pool = Pool(processes=4)
    t = dict()
    for i in range(4):
        t[i] = Test(i,10*i,100*i)
    
    argt = [t[0], t[1], t[2], t[3]]
    argx = [1.2, 2.2, 3.2, 4.2]
    argy = [-1.2, -2.2, -3.2, -4.2]
    #map(mod, argt, argx, argy)
    #pool.map(mod_unpack, argt, argx, argy)
    pool.map(mod_unpack, [(t[0], 1.2, -1.2), (t[1], 2.2, -2.2), (t[2], 3.2, -3.2), (t[3], 4.2, -4.2)] )
    #pool.map(mod_unpack, product(argt, argx, argy))
    for i in range(4):
        print t[i].x, t[i].y
