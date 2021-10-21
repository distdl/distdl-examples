import itertools
import copy

import numpy as np
import scipy
import scipy.linalg
from functools import reduce

__all__ = ['ImplicitSurface',
           'ImplicitCollection',
           'ImplicitPlane',
           'ImplicitSphere',
           'ImplicitXAlignedCylinder',
           'ImplicitEllipse',
           'ImplicitIntersection',
           'ImplicitUnion',
           'ImplicitDifference',
           'ImplicitComplement',
           'GridMapBase',
           'GridMap',
           'GridSlip'
           ]

class ImplicitSurface(object):

    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError('Must be implemented by subclass.')

    def interior(self, grid, asarray=False):
        val = self.__call__(grid)
        if asarray:
            retval = np.zeros_like(val)
            retval[np.where(val < 0.0)] = 1.0
            return retval
        else:
            return np.where(val < 0.0)

    def exterior(self, grid, asarray=False):
        val = self.__call__(grid)
        if asarray:
            retval = np.zeros_like(val)
            retval[np.where(val >= 0.0)] = 1.0
            return retval
        else:
            return np.where(val >= 0.0)

class ImplicitCollection(ImplicitSurface):

    def __init__(self, *items):
        if np.iterable(items[0]):
            self.items = list(items[0])
        else:
            self.items = list(items)

    def __call__(self):
        raise NotImplementedError('Must be implemented by subclass.')

class ImplicitPlane(ImplicitSurface):
    def __init__(self, p, n):
        self.p = np.array(p)
        self.n = np.array(n)
        self.n = self.n/np.linalg.norm(self.n)
        self.d = -np.dot(self.p,self.n)

    def __call__(self, grid):
        return self.d + reduce(lambda x,y:x+y, list(map(lambda x,y:x*y,self.n,grid)))

class ImplicitSphere(ImplicitSurface):
    def __init__(self, c=None, r=1.0):
        if c is None:
            self.c = None
        else:
            self.c = np.array(c)
        self.r = r

    def __call__(self, grid):
        if self.c is None:
            c = np.zeros_like(grid[0].shape)
        else:
            c = self.c
        return reduce(lambda x,y:x+y, list(map(lambda x,y:(y-x)**2,c,grid))) - self.r**2

class ImplicitXAlignedCylinder(ImplicitSurface):
    def __init__(self, c=None, length=1.0, r=1.0):
        if c is None:
            self.c = None
        else:
            self.c = np.array(c)
        self.len = length
        self.r = r

    def __call__(self, grid):
        if self.c is None:
            c = np.zeros_like(grid[0].shape)
        else:
            c = self.c

        g = grid[1:]
        cc = c[1:]
#       longways = (grid[1]-c[1])**2 - self.r**2
        longways =  reduce(lambda x,y:x+y, list(map(lambda x,y:(y-x)**2,cc,g))) - self.r**2
        cutoff   = np.abs(grid[0] - c[0]) -  self.len/2
        return np.maximum(longways, cutoff)

class ImplicitEllipse(ImplicitSurface):
    def __init__(self, c=None, a=None, r=1.0):
        if c is None:
            self.c = None
        else:
            self.c = np.array(c)
        if a is None:
            self.a = None
        else:
            self.a = np.array(a)
        self.r = r

    def __call__(self, grid):
        if self.c is None:
            c = np.zeros(len(grid))
        else:
            c = self.c
        if self.a is None:
            a = np.ones(len(grid))
        else:
            a = self.a
        return reduce(lambda x,y:x+y, list(map(lambda x,y,z:(((y-x)**2)/z), c,grid,a)) ) - self.r**2

class ImplicitIntersection(ImplicitCollection):
    def __init__(self, *items):
        ImplicitCollection.__init__(self, *items)

    def __call__(self, grid):
        return reduce(lambda x,y: np.maximum(x,y), [x(grid) for x in self.items])

class ImplicitUnion(ImplicitCollection):
    def __init__(self, *items):
        ImplicitCollection.__init__(self, *items)

    def __call__(self, grid):
        return reduce(lambda x,y: np.minimum(x,y), [x(grid) for x in self.items])

class ImplicitDifference(ImplicitCollection):

    # Maybe sometime, this should just take *items, and pop the first one for
    # base.  This way, we can allow a single list to be passed.  For now, whatever.
    def __init__(self, base, *items):
        ImplicitCollection.__init__(self, *items)
        self.base = base

    def __call__(self, grid):
        items = [self.base] + self.items
        return reduce(lambda x,y: np.maximum(x,-y), [x(grid) for x in items])

class ImplicitComplement(ImplicitSurface):

    def __init__(self, base):
        self.base = base

    def __call__(self, grid):
        return -1.0*self.base(grid)

    # These must be defined so that the equality is switched if the complement
    # is the calling surface
    def interior(self, grid, asarray=False):
        val = self.__call__(grid)
        if asarray:
            retval = np.zeros_like(val)
            retval[np.where(val <= 0.0)] = 1.0
            return retval
        else:
            return np.where(val <= 0.0)

    def exterior(self, grid, asarray=False):
        val = self.__call__(grid)
        if asarray:
            retval = np.zeros_like(val)
            retval[np.where(val > 0.0)] = 1.0
            return retval
        else:
            return np.where(val > 0.0)

class GridMapBase(object):
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError('Must be implemented by subclass.')


class GridMap(GridMapBase):
    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, grid):
        new_grid = []
        for f,g in zip(self.funcs,grid):
            if f is not None:
                new_grid.append(f(g))
            else:
                new_grid.append(g)

        return tuple(new_grid)

class GridSlip(GridMapBase):
    # Creates a slip or a fault along the specifid plane
    def __init__(self, p, n):
        self.p = np.array(p)
        self.n = np.array(n)
        self.n = self.n/np.linalg.norm(self.n)
        self.d = -np.dot(self.p,self.n)

        self.basis = None

    def __call__(self, grid, direction, amount):
        # amount is a scalar
        # direction is a vector that will be normalized
        # the slip occurs along that vector's projection onto the plane, i.e., it is orthogonal to the normal
        # the exterior of the slip plane (the part the normal points to) is the part that is modified.
        d = np.array(direction)
        d = d/np.linalg.norm(d)

        dim = len(d)

        if self.basis is None:
            basis = [self.n]

            # Gramm schmidt
            while len(basis) != dim:
                c = np.random.rand(3)
                for b in basis:
                    c -= np.dot(b,c)*b
                cn = np.linalg.norm(c)
                if cn > 1e-4:
                    basis.append(c/cn)

            self.basis = basis[1:]

        # Project the slip direction onto the basis
        proj=0
        for b in self.basis:
            proj += np.dot(d,b)*b

        # Scale the projection
        proj = amount*proj

        # Evaluate the plane to figure out what components of the old grid are shifted.
        val = self.d + reduce(lambda x,y:x+y, list(map(lambda x,y:x*y,self.n,grid)))
        loc = np.where(val >= 0.0)

        # Perform the shift.
        new_grid = [copy.deepcopy(g) for g in grid]
        for ng,p in zip(new_grid,proj):
            ng[loc] -=p

        return tuple(new_grid)

class Weird(ImplicitSurface):
    def __init__(self, p,n, c=None, r=1.0):
        if c is None:
            self.c = None
        else:
            self.c = np.array(c)
        self.r = r

        self.p = np.array(p)
        self.n = np.array(n)
        self.n = self.n/np.linalg.norm(self.n)
        self.d = -np.dot(self.p,self.n)

    def __call__(self, grid):
        if self.c is None:
            c = np.zeros_like(grid[0].shape)
        else:
            c = self.c

# Creates flat eye thingies
#       val1 = reduce(lambda x,y:x+y, map(lambda x,y:(y-x)**2,c,grid)) - self.r**2
#
#       val2 = self.d + reduce(lambda x,y:x+y, map(lambda x,y:x*y,self.n,grid))
#
#       return 4*val1/np.max(val1)+val2

        val1 = reduce(lambda x,y:x+y, list(map(lambda x,y:(y-x)**2,c,grid))) - self.r**2

        val2 = self.d + reduce(lambda x,y:x+y, list(map(lambda x,y:x*y,self.n,grid)))

        return 4*val1/np.max(val1)+val2

class Hyperbola(ImplicitSurface):
    def __init__(self, c=None, r=1.0):
        if c is None:
            self.c = None
        else:
            self.c = np.array(c)
        self.r = r

    def __call__(self, grid):
        if self.c is None:
            c = np.zeros_like(grid[0].shape)
        else:
            c = self.c

        return  reduce(lambda x,y:-x+y, list(map(lambda x,y:(y-x)**2,c,grid))) - self.r**2

class Weird2(ImplicitSurface):
    def __init__(self, c=None, r=1.0, s=None):
        if c is None:
            self.c = None
        else:
            self.c = np.array(c)

        if s is None:
            self.s = None
        else:
            self.s = np.array(s)

        self.r = r

    def __call__(self, grid):
        c = np.zeros_like(grid[0].shape) if self.c is None else self.c
        s = np.ones_like(grid[0].shape) if self.s is None else self.s

        return ((grid[0]-c[0])**2)/s[0] + ((grid[1]-c[1])**1)/s[1] - self.r**2
        #reduce(lambda x,y:-x+y, map(lambda x,y:(y-x)**2,c,grid)) - self.r**2

# FIXME
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    from PIL import Image

    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 64)
    z = np.linspace(0, 1, 64)

    grid = np.meshgrid(x, y, z)

    shape1 = ImplicitEllipse(c=(0.5, 0.5), a=(2.0, 1.0), r=0.25)
    value1 = shape1(grid)
    segmentation1 = shape1.interior(grid, True)
    image1 = -1*value1*segmentation1

    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(value1)
    # plt.colorbar()
    # plt.subplot(1,3,2)
    # plt.imshow(segmentation1)
    # plt.colorbar()
    # plt.subplot(1,3,3)
    # plt.imshow(image1)
    # plt.colorbar()


    def random_ellipses(n, d, r_shift=0.3, r_fac=0.01):

        Es = list()

        for i in range(n):

            c = np.random.rand(d)
            a = 1.5*np.random.rand(d)
            # a = np.ones(d)
            r = r_shift + r_fac*np.random.rand(1)
            # r = 0.2

            Es.append(ImplicitEllipse(c, a, r))

        return ImplicitUnion(*Es)

    # --------------------

    # n_ellipses = 5
    # dim = 2

    # shape2 = random_ellipses(n_ellipses, dim, r_shift=0.2)
    # value2 = shape2(grid)
    # segmentation2 = shape2.interior(grid, True)
    # image2 = -1*value2*segmentation2

    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(value2)
    # plt.colorbar()
    # plt.subplot(1,3,2)
    # plt.imshow(segmentation2)
    # plt.colorbar()
    # plt.subplot(1,3,3)
    # plt.imshow(image2)
    # plt.colorbar()

    # --------------------

    n_ellipses_target = 3
    n_ellipses_noise = 2
    dim = 3

    shape_target = random_ellipses(n_ellipses_target, dim)
    value_target = shape_target(grid)
    segmentation_target = shape_target.interior(grid, True)
    image_target = -1*value_target*segmentation_target

    shape_noise = random_ellipses(n_ellipses_noise, dim, 0.2, 0.1)
    value_noise = shape_noise(grid)
    segmentation_noise = shape_noise.interior(grid, True)
    image_noise = -1*value_noise*segmentation_noise

    image_blended = image_target + 0.3*image_noise

    img = Image.fromarray(image_blended,"RGB")
    target = Image.fromarray(segmentation_target,"RGB")
    print(segmentation_target.shape)

    """ plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(image_target)
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(segmentation_target)
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(image_blended)
    plt.colorbar()

    plt.show() """
