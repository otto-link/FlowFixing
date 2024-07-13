# Copyright (c) 2024 Otto Link. Distributed under the terms of the GNU
# General Public License. The full license is in the file LICENSE,
# distributed with this software.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy

import tools.flow_fixing as ff

if __name__ == '__main__':

    # load heightmap
    fname = 'data/hmap.png'
    # fname = 'data/hmap_high_kw.png' # with much more bumps and sinks
    # fname = 'data/hmap_bump.png'
    fname = 'data/hmap_hole.png'
    
    z = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    z = ff.remap(z)

    # apply flow fixing
    z_fixed = np.copy(z)
    ff.flow_fixing(z_fixed,
                   gaussian_filter_sigma=2,
                   riverbed_talus=0.1 / z.shape[0],
                   plot_intermediate_step=True)

    # make something a bit smoother
    z_riverbed = ff.carve_river(z,
                                z_fixed,
                                riverbank_talus=8.0 / z.shape[0],
                                merging_distance=4)

    # save as grayscale png
    z_riverbed = ff.remap(z_riverbed)
    img = np.array(z_riverbed * 255).astype('uint8')
    cv2.imwrite('hmap_fixed.png', img)

    plt.figure()
    plt.imshow(z.T, origin='lower', cmap='terrain')

    plt.figure()
    plt.imshow(z_fixed.T, origin='lower', cmap='terrain')

    plt.figure()
    plt.imshow(z_riverbed.T, origin='lower', cmap='terrain')

    plt.show()
