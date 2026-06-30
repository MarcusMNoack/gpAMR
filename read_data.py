#
# C. Paolini
# paolini@engineering.sdsu.edu
# LBNL 06/14/23
#
import sys
import os
import h5py
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
import itertools

# # Get the filename from the command line
# def make_block_data(x_data, y_data, M, N, m, n):
#     points = x_data
#     y = y_data
#     grid = points.reshape(M, N, 2)
#     y_grid = y.reshape(M, N)
#     plt.imshow(y_grid)
#     plt.show()


#     point_patches = []
#     y_patches = []
    
#     for i in range(0, N - n + 1, n):
#         for j in range(0, M - m + 1, m):
#             grid_shape = grid[i:i+n+1, j:j+m+1, :].shape  # (n+1, m+1, 2)
#             point_patch = grid[i:i+n+1, j:j+m+1, :].reshape(grid_shape[0] * grid_shape[1], 2)
#             y_patch = y_grid[i:i+n+1, j:j+m+1].reshape(grid_shape[0] * grid_shape[1])
#             point_patches.append(point_patch)
#             y_patches.append(y_patch)
#     return point_patches, y_patches



# def read_hdf5(filename, index = "vorticity"):
#     if os.path.isfile(filename):
#         hf_in = h5py.File(filename,"r")
#         root = hf_in["/"]
#         data = hf_in["level_0/data:datatype=0"]
#         offsets = hf_in["level_0/data:offsets=0"]
#         attributes = hf_in["level_0/data_attributes/"]
#         boxes = hf_in["level_0/boxes"]
#         nBoxes = boxes.shape[0]
#         boxDim = (boxes[0][2] - boxes[0][0] + 1, boxes[0][3] - boxes[0][1] + 1)
#         print(f'box dimension: {boxDim}')
#         comps = attributes.attrs["comps"]
#         print(f'components: {comps}')

#         components = [i.decode('utf-8') if isinstance(i, np.bytes_) else '' for i in list(root.attrs.values())]

#         level_0 = hf_in["level_0/"]
#         prob_domain = level_0.attrs["prob_domain"]
#         time = level_0.attrs["time"]
#         dataNumPy = np.array(data, np.float64)
#         boxData = dataNumPy.reshape((nBoxes,int(dataNumPy.shape[0]/nBoxes))) ##box data is the data by-box
#         print(f'data dimension: {boxData.shape}')

#         X, Y = np.mgrid[prob_domain[1]:prob_domain[3]+1, prob_domain[0]:prob_domain[2]+1]
#         nRows = X.shape[0]
#         nCols = X.shape[1]

#         patchRows = int(nRows/boxDim[1])
#         patchCols = int(nCols/boxDim[0])

#         print(f'grid dimension: {X.shape}')
#         print(f'patch columns: {patchCols}')
#         print(f'patch rows: {patchRows}')

#         component0_i = components.index(index)
#         componentKeys = list(root.attrs.keys())
#         componentValues = list(root.attrs.values())
#         component0_i = int(re.findall(r'\d+', componentKeys[component0_i])[0]) 
#         component0 = np.zeros(X.shape)
#         Block_dict= {}
#         global_x = np.arange(prob_domain[0],prob_domain[2]+1)
#         global_y = np.arange(prob_domain[1],prob_domain[3]+1)
#         global_xy = np.asarray(list(itertools.product(global_x,global_y)))
#         for col in range(patchCols):
#             for row in range(patchRows):
#                 # Extract the flat block
#                 flat_block = boxData[
#                     col * patchRows + row,
#                     component0_i * (boxDim[0] * boxDim[1]) : (component0_i + 1) * (boxDim[0] * boxDim[1])
#                 ]
#                 domain_x_extension = 1 if col<patchCols-1 else 0
#                 domain_y_extension = 1 if row<patchRows-1 else 0

#                 x_range = (row * boxDim[0] , (row + 1) * boxDim[0])
#                 y_range = (col * boxDim[1] , (col + 1) * boxDim[1])
#                 x = np.arange(y_range[0], y_range[1])
#                 y = np.arange(x_range[0], x_range[1])
#                 xy = np.asarray(list(itertools.product(x,y)))
        
#                 # Reshape it into (boxDim[0], boxDim[1])
#                 block = flat_block.reshape(boxDim[0], boxDim[1])
#                 ## Assign into the big array
#                 component0[
#                     row * boxDim[0] : (row + 1) * boxDim[0],
#                     col * boxDim[1] : (col + 1) * boxDim[1]
#                 ] = block
#                 Block_dict[str(row)+","+str(col)] = (xy, block.T.flatten(), np.array([[y_range[0], y_range[1]-1],[x_range[0], x_range[1]-1]]))
                
#         prob_domain = np.asarray([float(entry) for entry in prob_domain])
#         prob_domain = prob_domain.reshape(2,2).T
#         res_dict = {
#                     "global x": global_xy,
#                     "global y": component0.T.reshape(component0.size),
#                     "domain": prob_domain,
#                     "block data": Block_dict}
#         return res_dict

#     else:
#         return None, None


# #"component0" for the advection problem

# def read_hdf5II(filename, index = "vorticity"):
#     if os.path.isfile(filename):
#         hf_in = h5py.File(filename,"r")
#         root = hf_in["/"]
#         data = hf_in["level_0/data:datatype=0"]
#         offsets = hf_in["level_0/data:offsets=0"]
#         attributes = hf_in["level_0/data_attributes/"]
#         boxes = hf_in["level_0/boxes"]
#         nBoxes = boxes.shape[0]
#         boxDim = (boxes[0][2] - boxes[0][0] + 1, boxes[0][3] - boxes[0][1] + 1)
#         print(f'box dimension: {boxDim}')
#         comps = attributes.attrs["comps"]
#         print(f'components: {comps}')

#         components = [i.decode('utf-8') if isinstance(i, np.bytes_) else '' for i in list(root.attrs.values())]
#         print(components)

#         level_0 = hf_in["level_0/"]
#         prob_domain = level_0.attrs["prob_domain"]
#         time = level_0.attrs["time"]
#         dataNumPy = np.array(data, np.float64)
#         boxData = dataNumPy.reshape((nBoxes,int(dataNumPy.shape[0]/nBoxes))) ##box data is the data by-box
#         print(f'data dimension: {boxData.shape}')

#         X, Y = np.mgrid[prob_domain[1]:prob_domain[3]+1, prob_domain[0]:prob_domain[2]+1]
#         nRows = X.shape[0]
#         nCols = X.shape[1]

#         patchRows = int(nRows/boxDim[1])
#         patchCols = int(nCols/boxDim[0])

#         print(f'grid dimension: {X.shape}')
#         print(f'patch columns: {patchCols}')
#         print(f'patch rows: {patchRows}')

#         component0_i = components.index(index)
#         componentKeys = list(root.attrs.keys())
#         componentValues = list(root.attrs.values())
#         component0_i = int(re.findall(r'\d+', componentKeys[component0_i])[0]) 
#         component0 = np.zeros(X.shape)
#         Block_dict= {}
#         global_x = np.arange(prob_domain[0],prob_domain[2]+1)
#         global_y = np.arange(prob_domain[1],prob_domain[3]+1)
#         global_xy = np.asarray(list(itertools.product(global_x,global_y)))
#         for col in range(patchCols):
#             for row in range(patchRows):
#                 # Extract the flat block
#                 flat_block = boxData[
#                     col * patchRows + row,
#                     component0_i * (boxDim[0] * boxDim[1]) : (component0_i + 1) * (boxDim[0] * boxDim[1])
#                 ]
#                 x_range = (row * boxDim[0] , (row + 1) * boxDim[0])
#                 y_range = (col * boxDim[1] , (col + 1) * boxDim[1])
#                 x = np.arange(y_range[0], y_range[1])
#                 y = np.arange(x_range[0], x_range[1])
#                 xy = np.asarray(list(itertools.product(x,y)))
        
#                 # Reshape it into (boxDim[0], boxDim[1])
#                 block = flat_block.reshape(boxDim[0], boxDim[1])
        
#                 ## Assign into the big array
#                 component0[
#                     row * boxDim[0] : (row + 1) * boxDim[0],
#                     col * boxDim[1] : (col + 1) * boxDim[1]
#                 ] = block
                
#         prob_domain = np.asarray([float(entry) for entry in prob_domain])
#         prob_domain = prob_domain.reshape(2,2).T
#         y_data = component0.T.reshape(component0.size)
#         res_dict = {
#                     "global x": global_xy,
#                     "global y": y_data,
#                     "domain": prob_domain}
#         print("max(y_data): ", np.max(y_data), " min(y_data): ", np.min(y_data))
#         return res_dict

#     else:
#         return None, None


import os
import re
import itertools

import h5py
import numpy as np


def read_hdf5_decomposed(filename, index="vorticity",
                         n_sub_x=1, n_sub_y=1, pad_x=0, pad_y=0):
    """
    Read a single-level Chombo HDF5 dump, assemble one component into the full
    global field, and decompose that field into n_sub_y x n_sub_x subdomains
    with halo padding.

    Padding rule for each subdomain:
      - ghost cells are filled with the *actual* values of the neighbouring
        subdomain (they are just a wider slice of the same global field),
      - where a subdomain edge coincides with the global domain edge the halo
        is clipped, so no padding is added there.
    `pad_x` / `pad_y` set the halo width (in cells) across the column and row
    cuts independently.

    Returns
    -------
    dict with keys:
      "global_funcvalues"  : (M,)   field values, x-slow / y-fast order
      "global_coordinates" : (M, 2) matching (x, y) coordinates
      "local_data"         : {id: {...}} one padded subdomain per integer id
      "domain"             : (2, 2) prob_domain as [[x0,x1],[y0,y1]]
      "time"               : float simulation time
    Each local_data entry holds:
      "points"    : (Li, 2) padded coordinates (interior + halo)
      "values"    : (Li,)   padded field values
      "interior"  : (Li,)   bool mask, True = owned cell, False = ghost
      "grid_pos"  : (sy, sx) subdomain position in the decomposition grid
      "bounds"    : interior / padded index ranges into the global field
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    with h5py.File(filename, "r") as hf_in:
        root = hf_in["/"]
        data = hf_in["level_0/data:datatype=0"]
        attributes = hf_in["level_0/data_attributes/"]
        boxes = hf_in["level_0/boxes"]
        level_0 = hf_in["level_0/"]

        nBoxes = boxes.shape[0]
        boxDim = (boxes[0][2] - boxes[0][0] + 1,
                  boxes[0][3] - boxes[0][1] + 1)
        prob_domain = level_0.attrs["prob_domain"]
        time = float(level_0.attrs["time"])

        # component name -> attribute key ("component_3") -> integer offset 3
        components = [v.decode("utf-8") if isinstance(v, np.bytes_) else ""
                      for v in list(root.attrs.values())]
        comp_name_i = components.index(index)
        comp_key = list(root.attrs.keys())[comp_name_i]
        comp_i = int(re.findall(r"\d+", comp_key)[0])

        dataNumPy = np.array(data, np.float64)
        boxData = dataNumPy.reshape((nBoxes, dataNumPy.shape[0] // nBoxes))

    # ---- assemble the global field (kept identical to the original tiling) --
    nx = int(prob_domain[2] - prob_domain[0] + 1)   # columns (x)
    ny = int(prob_domain[3] - prob_domain[1] + 1)   # rows    (y)
    field = np.zeros((ny, nx))                       # field[row=y, col=x]

    patchCols = nx // boxDim[0]
    patchRows = ny // boxDim[1]
    span = boxDim[0] * boxDim[1]

    for col in range(patchCols):
        for row in range(patchRows):
            flat_block = boxData[col * patchRows + row,
                                 comp_i * span: (comp_i + 1) * span]
            block = flat_block.reshape(boxDim[0], boxDim[1])
            field[row * boxDim[0]: (row + 1) * boxDim[0],
                  col * boxDim[1]: (col + 1) * boxDim[1]] = block

    # ---- global coordinates / values (x slowest, y fastest) ----------------
    gx = np.arange(prob_domain[0], prob_domain[2] + 1)   # (nx,)
    gy = np.arange(prob_domain[1], prob_domain[3] + 1)   # (ny,)
    global_coordinates = np.asarray(list(itertools.product(gx, gy)))
    global_funcvalues = field.T.reshape(field.size)      # matches product order

    # ---- halo decomposition ------------------------------------------------
    col_tiles = np.array_split(np.arange(nx), n_sub_x)   # balanced, contiguous
    row_tiles = np.array_split(np.arange(ny), n_sub_y)

    local_data = {}
    sid = 0
    for sy, ry in enumerate(row_tiles):
        r0, r1 = int(ry[0]), int(ry[-1] + 1)             # interior rows [r0, r1)
        for sx, cx in enumerate(col_tiles):
            c0, c1 = int(cx[0]), int(cx[-1] + 1)         # interior cols [c0, c1)

            # clip halo at the global edge -> no padding on a global boundary
            pr0, pr1 = max(0, r0 - pad_y), min(ny, r1 + pad_y)
            pc0, pc1 = max(0, c0 - pad_x), min(nx, c1 + pad_x)

            block = field[pr0:pr1, pc0:pc1]              # halo = neighbour data
            xx, yy = np.meshgrid(gx[pc0:pc1], gy[pr0:pr1])  # (rows, cols)

            interior = np.zeros(block.shape, bool)
            interior[r0 - pr0: r1 - pr0, c0 - pc0: c1 - pc0] = True

            local_data[sid] = {
                "points":   np.column_stack([xx.ravel(), yy.ravel()]),
                "values":   block.ravel(),
                "interior": interior.ravel(),
                "grid_pos": (sy, sx),
                "bounds": {
                    "interior_rows": (r0, r1), "interior_cols": (c0, c1),
                    "padded_rows":   (pr0, pr1), "padded_cols":   (pc0, pc1),
                },
            }
            sid += 1

    prob_domain = np.asarray([float(e) for e in prob_domain]).reshape(2, 2).T

    return {
        "global_funcvalues":  global_funcvalues,
        "global_coordinates": global_coordinates,
        "local_data":         local_data,
        "domain":             prob_domain,
        "time":               time,
    }


def reassemble(result):
    """
    Stitch subdomain values back into the global field order using only the
    interior (owned) cells, so overlapping halos never write the same cell
    twice. Returns a (M,) array aligned with result["global_coordinates"].
    """
    coords = result["global_coordinates"]
    out = np.full(len(coords), np.nan)

    # map global (x, y) -> flat index in the x-slow / y-fast ordering
    lookup = {tuple(xy): k for k, xy in enumerate(map(tuple, coords))}
    for sub in result["local_data"].values():
        m = sub["interior"]
        for xy, v in zip(map(tuple, sub["points"][m]), sub["values"][m]):
            out[lookup[xy]] = v
    return out
