import numpy as np
import torch

from implicit import ImplicitEllipse,ImplicitUnion

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

def gen_data(grid, n_ellipses, n_noise):
    n_ellipses_target = n_ellipses
    n_ellipses_noise = n_noise
    dim = len(grid)

    shape_target = random_ellipses(n_ellipses_target, dim)

    value_target = shape_target(grid)
    segmentation_target = shape_target.interior(grid, True)
    image_target = -1*value_target*segmentation_target

    shape_noise = random_ellipses(n_ellipses_noise, dim, 0.2, 0.1)
    value_noise = shape_noise(grid)

    segmentation_noise = shape_noise.interior(grid, True)
    image_noise = -1*value_noise*segmentation_noise

    image_blended = image_target + 0.3*image_noise


    unsqueeze = lambda x: torch.unsqueeze(torch.unsqueeze(x, 0), 0)

    img = unsqueeze(torch.from_numpy(image_blended).float())
    mask = unsqueeze(torch.from_numpy(segmentation_target).float())

    return img, mask