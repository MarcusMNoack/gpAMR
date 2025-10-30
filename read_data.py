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

# Get the filename from the command line
def make_block_data(x_data, y_data, M, N, m, n):
    points = x_data
    y = y_data
    grid = points.reshape(M, N, 2)
    y_grid = y.reshape(M, N)
    plt.imshow(y_grid)
    plt.show()


    point_patches = []
    y_patches = []
    
    for i in range(0, N - n + 1, n):
        for j in range(0, M - m + 1, m):
            grid_shape = grid[i:i+n+1, j:j+m+1, :].shape  # (n+1, m+1, 2)
            point_patch = grid[i:i+n+1, j:j+m+1, :].reshape(grid_shape[0] * grid_shape[1], 2)
            y_patch = y_grid[i:i+n+1, j:j+m+1].reshape(grid_shape[0] * grid_shape[1])
            point_patches.append(point_patch)
            y_patches.append(y_patch)
    return point_patches, y_patches



def read_hdf5(filename, index = "vorticity"):
    if os.path.isfile(filename):
        hf_in = h5py.File(filename,"r")
        root = hf_in["/"]
        data = hf_in["level_0/data:datatype=0"]
        offsets = hf_in["level_0/data:offsets=0"]
        attributes = hf_in["level_0/data_attributes/"]
        boxes = hf_in["level_0/boxes"]
        nBoxes = boxes.shape[0]
        boxDim = (boxes[0][2] - boxes[0][0] + 1, boxes[0][3] - boxes[0][1] + 1)
        print(f'box dimension: {boxDim}')
        comps = attributes.attrs["comps"]
        print(f'components: {comps}')

        components = [i.decode('utf-8') if isinstance(i, np.bytes_) else '' for i in list(root.attrs.values())]

        level_0 = hf_in["level_0/"]
        prob_domain = level_0.attrs["prob_domain"]
        time = level_0.attrs["time"]
        dataNumPy = np.array(data, np.float64)
        boxData = dataNumPy.reshape((nBoxes,int(dataNumPy.shape[0]/nBoxes))) ##box data is the data by-box
        print(f'data dimension: {boxData.shape}')

        X, Y = np.mgrid[prob_domain[1]:prob_domain[3]+1, prob_domain[0]:prob_domain[2]+1]
        nRows = X.shape[0]
        nCols = X.shape[1]

        patchRows = int(nRows/boxDim[1])
        patchCols = int(nCols/boxDim[0])

        print(f'grid dimension: {X.shape}')
        print(f'patch columns: {patchCols}')
        print(f'patch rows: {patchRows}')

        component0_i = components.index(index)
        componentKeys = list(root.attrs.keys())
        componentValues = list(root.attrs.values())
        component0_i = int(re.findall(r'\d+', componentKeys[component0_i])[0]) 
        component0 = np.zeros(X.shape)
        Block_dict= {}
        global_x = np.arange(prob_domain[0],prob_domain[2]+1)
        global_y = np.arange(prob_domain[1],prob_domain[3]+1)
        global_xy = np.asarray(list(itertools.product(global_x,global_y)))
        for col in range(patchCols):
            for row in range(patchRows):
                # Extract the flat block
                flat_block = boxData[
                    col * patchRows + row,
                    component0_i * (boxDim[0] * boxDim[1]) : (component0_i + 1) * (boxDim[0] * boxDim[1])
                ]
                x_range = (row * boxDim[0] , (row + 1) * boxDim[0])
                y_range = (col * boxDim[1] , (col + 1) * boxDim[1])
                x = np.arange(y_range[0], y_range[1])
                y = np.arange(x_range[0], x_range[1])
                xy = np.asarray(list(itertools.product(x,y)))
        
                # Reshape it into (boxDim[0], boxDim[1])
                block = flat_block.reshape(boxDim[0], boxDim[1])
        
                ## Assign into the big array
                component0[
                    row * boxDim[0] : (row + 1) * boxDim[0],
                    col * boxDim[1] : (col + 1) * boxDim[1]
                ] = block
                Block_dict[str(row)+","+str(col)] = (xy, block.T.flatten(), np.array([[y_range[0], y_range[1]],[x_range[0], x_range[1]]]))
                
        prob_domain = np.asarray([float(entry) for entry in prob_domain])
        prob_domain = prob_domain.reshape(2,2).T
        res_dict = {
                    "global x": global_xy,
                    "global y": component0.T.reshape(component0.size),
                    "domain": prob_domain,
                    "block data": Block_dict}
        return res_dict

    else:
        return None, None


def read_hdf5II(filename, index = "vorticity"):
    if os.path.isfile(filename):
        hf_in = h5py.File(filename,"r")
        root = hf_in["/"]
        data = hf_in["level_0/data:datatype=0"]
        offsets = hf_in["level_0/data:offsets=0"]
        attributes = hf_in["level_0/data_attributes/"]
        boxes = hf_in["level_0/boxes"]
        nBoxes = boxes.shape[0]
        boxDim = (boxes[0][2] - boxes[0][0] + 1, boxes[0][3] - boxes[0][1] + 1)
        print(f'box dimension: {boxDim}')
        comps = attributes.attrs["comps"]
        print(f'components: {comps}')

        components = [i.decode('utf-8') if isinstance(i, np.bytes_) else '' for i in list(root.attrs.values())]

        level_0 = hf_in["level_0/"]
        prob_domain = level_0.attrs["prob_domain"]
        time = level_0.attrs["time"]
        dataNumPy = np.array(data, np.float64)
        boxData = dataNumPy.reshape((nBoxes,int(dataNumPy.shape[0]/nBoxes))) ##box data is the data by-box
        print(f'data dimension: {boxData.shape}')

        X, Y = np.mgrid[prob_domain[1]:prob_domain[3]+1, prob_domain[0]:prob_domain[2]+1]
        nRows = X.shape[0]
        nCols = X.shape[1]

        patchRows = int(nRows/boxDim[1])
        patchCols = int(nCols/boxDim[0])

        print(f'grid dimension: {X.shape}')
        print(f'patch columns: {patchCols}')
        print(f'patch rows: {patchRows}')

        component0_i = components.index(index)
        componentKeys = list(root.attrs.keys())
        componentValues = list(root.attrs.values())
        component0_i = int(re.findall(r'\d+', componentKeys[component0_i])[0]) 
        component0 = np.zeros(X.shape)
        Block_dict= {}
        global_x = np.arange(prob_domain[0],prob_domain[2]+1)
        global_y = np.arange(prob_domain[1],prob_domain[3]+1)
        global_xy = np.asarray(list(itertools.product(global_x,global_y)))
        for col in range(patchCols):
            for row in range(patchRows):
                # Extract the flat block
                flat_block = boxData[
                    col * patchRows + row,
                    component0_i * (boxDim[0] * boxDim[1]) : (component0_i + 1) * (boxDim[0] * boxDim[1])
                ]
                x_range = (row * boxDim[0] , (row + 1) * boxDim[0])
                y_range = (col * boxDim[1] , (col + 1) * boxDim[1])
                x = np.arange(y_range[0], y_range[1])
                y = np.arange(x_range[0], x_range[1])
                xy = np.asarray(list(itertools.product(x,y)))
        
                # Reshape it into (boxDim[0], boxDim[1])
                block = flat_block.reshape(boxDim[0], boxDim[1])
        
                ## Assign into the big array
                component0[
                    row * boxDim[0] : (row + 1) * boxDim[0],
                    col * boxDim[1] : (col + 1) * boxDim[1]
                ] = block
                
        prob_domain = np.asarray([float(entry) for entry in prob_domain])
        prob_domain = prob_domain.reshape(2,2).T
        y_data = component0.T.reshape(component0.size)
        res_dict = {
                    "global x": global_xy,
                    "global y": y_data,
                    "domain": prob_domain}
        print("max(y_data): ", np.max(y_data), " min(y_data): ", np.min(y_data))
        return res_dict

    else:
        return None, None