# Copyright (c) 2024 Otto Link. Distributed under the terms of the GNU
# General Public License. The full license is in the file LICENSE,
# distributed with this software.
import heapq
import numpy as np
import scipy
import sys

import matplotlib.pyplot as plt

# neighbor search
di = [1, 1, 0, -1, -1, -1, 0, 1]
dj = [0, 1, 1, 1, 0, -1, -1, -1]


def is_in_inner_domain(i, j, shape):
    return (i > 0) and (j > 0) and (i < shape[0] - 1) and (j < shape[1] - 1)


def min_over_path(z, path):
    zmin = sys.float_info.max
    for p in path:
        zmin = min(zmin, z[p])
    return zmin


def find_sinks(z, gaussian_filter_sigma):
    z_filtered = scipy.ndimage.gaussian_filter(z, gaussian_filter_sigma)
    z_minf = scipy.ndimage.minimum_filter(z_filtered, size=(3, 3))

    # if the elevation of a cell is the lowest elevation compared to
    # its first neighbors, then it's a sink, since flow cannot go out
    # from this cell
    is_sink = np.where(z_filtered == z_minf, True, False)

    return is_sink


def flow_fixing(z, gaussian_filter_sigma, riverbed_talus, max_iterations=10, plot_intermediate_step=False):

    it = 0
    nb_of_sinks = 0

    # for flats rivers (no downslope), the river system can be built
    # in only one pass. If not, an iterative process seems to be
    # necessary
    if (riverbed_talus == 0.0):
        max_iterations = 1

    while (it < max_iterations):

        if (plot_intermediate_step):
            plt.figure()
            plt.imshow(z.T, origin='lower', cmap='gray', vmin=0, vmax=1)
            plt.gca().axis('off')

        # --- find the sinks and use the number of sinks as a stop
        # --- criteria for the algo (if this number does not change
        # --- after an iteration, no need to continue...)

        is_sink = find_sinks(z, gaussian_filter_sigma)
        is_sink_unfiltered = find_sinks(z, 0)

        is_sink = np.logical_and(is_sink, is_sink_unfiltered)

        new_nb_of_sinks = np.count_nonzero(is_sink[1:-1, 1:-1])

        if (new_nb_of_sinks == nb_of_sinks):
            break
        else:
            nb_of_sinks = new_nb_of_sinks

        print('iteration #{}'.format(it + 1))
            
        # --- flow breaching: 1st pass

        # initialize heap queue: algo starts from the lowest cells on the domain border
        qz = []
        for i0 in [0, z.shape[0] - 1]:
            j0 = np.argmin(z[i0, :])
            qz.append([z[i0, j0], (i0, j0)])
        heapq.heapify(qz)

        is_done = np.zeros(z.shape, dtype=bool)

        flow_map = {}
        breach_history = {}

        while (len(qz)):
            _, (i, j) = heapq.heappop(qz)

            # check neighbor cells for flow sinks, if not keep searching and follow
            # the lowest elevation path (using the heap queue)
            for k in range(8):

                ik = min(max(0, i + di[k]), z.shape[0] - 1)
                jk = min(max(0, j + dj[k]), z.shape[0] - 1)

                if not (is_done[ik, jk]):
                    heapq.heappush(qz, [z[ik, jk], (ik, jk)])
                   
                    # store flow direction
                    flow_map[(ik, jk)] = (i, j)
                    is_done[ik, jk] = True

                    # if the current cell is a sink, "breach" the heightmap by
                    # following the reverse flow direction in order to connect this
                    # sink to another sink, or to connect this connect to the
                    # domain border
                    if (is_sink[ik, jk]):

                        if (plot_intermediate_step):
                            plt.plot(ik, jk, 'bo')
                      
                        ib = ik
                        jb = jk

                        keep_breaching = True
                        path = [(ib, jb)]

                        while (keep_breaching
                               and is_in_inner_domain(ib, jb, z.shape)):
                            ib, jb = flow_map[(ib, jb)]
                            path.append((ib, jb))

                            if (is_sink[ib, jb]):
                                keep_breaching = False

                        # after the breaching path has been identifier, follow the
                        # this path and make sure the elevation is monotonic along
                        # this path

                        # also store all the breaching path for the second pass of
                        # the algorithm
                        if path:
                            if (path[0] != path[-1]):
                                breach_history[path[0] + path[-1]] = path

                                zmin = min(z[path[-1]], z[path[0]])
                                for p, pnext in zip(path[:-1], path[1:]):
                                    if z[pnext] > z[p]:
                                        z[pnext] = z[p] - riverbed_talus

                                if (plot_intermediate_step):
                                    x = [v[0] for v in path]
                                    y = [v[1] for v in path]
                                    plt.plot(x, y, 'b.', ms=0.5)


        # --- second pass: breach again

        # find min elevation for each end node of the path
        zmin_dict = {}

        for idx1 in breach_history.keys():
            for ij in [idx1[2:4]]:
                zmin_dict[ij] = 1e9
                for idx2, path in breach_history.items():
                    if (ij == idx2[0:2]) or (ij == idx2[2:4]):
                        zmin_dict[ij] = min(zmin_dict[ij],
                                            min_over_path(z, path))

        # breach again to ensure overall elevations are coherent between the sinks
        pq = [(-z[idx[2:4]], idx) for idx in breach_history.keys()]
        heapq.heapify(pq)

        while (pq):
            _, idx = heapq.heappop(pq)

            zmin = zmin_dict[idx[2:4]]
            path = breach_history[idx]

            # second pass, ensure downslope
            for p, pnext in zip(path[:-1], path[1:]):
                if z[pnext] > z[p]:
                    z[pnext] = z[p] - riverbed_talus

        # next one
        it += 1

    return None


def expand_talus(z, mask, talus):

    idx = np.where(mask)

    # initialize heap queue: algo starts from the cells defined by the
    # mask
    qz = []
    for i, j in zip(idx[0], idx[1]):
        qz.append([z[i, j], (i, j)])
    heapq.heapify(qz)

    is_done = np.copy(mask)

    while (len(qz)):
        _, (i, j) = heapq.heappop(qz)

        for p, q in zip(di, dj):
            ib = i + p
            jb = j + q

            if (is_in_inner_domain(ib, jb, z.shape)):
                z[ib, jb] = min(z[ib, jb], z[i, j] + talus)

                if not (is_done[ib, jb]):
                    heapq.heappush(qz, [z[ib, jb], (ib, jb)])
                    is_done[ib, jb] = True

    return z


def carve_river(z_initial, z_flow_fixed, riverbank_talus, merging_distance):

    # where modifications have been made
    mask = np.where(z_initial != z_flow_fixed, True, False)

    z_riverbed = np.copy(z_flow_fixed)
    z_riverbed = expand_talus(z_riverbed, mask, talus=riverbank_talus)

    # add a bit of smoothing
    z_riverbed = scipy.ndimage.gaussian_filter(z_riverbed, 1)

    # use a distance transform to define a merging mask between the
    # input heightmap "z_initial" and the smooth flow fixed heighmap
    # "z_riverbed" (NB - merging_distance in pixels)
    dist = scipy.ndimage.distance_transform_edt(1 - mask)
    dist = np.exp(-0.5 * dist**2 / merging_distance**2)

    # lerp based on distance
    z_riverbed = dist * z_riverbed + (1 - dist) * z_initial

    return z_riverbed


def remap(z):
    return (z - z.min()) / z.ptp()
