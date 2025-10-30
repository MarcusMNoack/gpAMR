import csv
import sys
import numpy as np
import os
import ast
import random
import time
import pickle
import gc
from read_data import *
from datetime import datetime
import shutil
from functools import partial
import dask
from dask.distributed import Client
import os
import time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata



def make_block_data(x_data, y_data, M, N, m, n):
    points = x_data
    y = y_data
    grid = points.reshape(M, N, 2)
    y_grid = y.reshape(M, N)


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


def trim_array(arr, max_len=10000):
    n = len(arr)
    if n <= max_len:
        return arr
    # Compute how many elements to remove
    excess = n - max_len
    # Remove roughly every nth element
    step = n // excess
    indices_to_remove = np.arange(0, n, step)[:excess]
    return np.delete(arr, indices_to_remove)



def init_client(scheduler_file, n_workers):
    while True:
        time.sleep(1)
        if os.path.isfile(scheduler_file):
            print("file found")
            time.sleep(2)
            try: client = Client(scheduler_file=scheduler_file)
            except: continue
            break
    print("waiting for workers")
    client.wait_for_workers(n_workers)
    workers = client.scheduler_info(n_workers = -1)["workers"]
    print("Number of availible workers: ", len(workers))
    return client


######################################
######################################
def chunks(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

######################################
######################################

def tell(client, x_data, y_data, noise,GP):
    future = client.submit(lambda obj: obj.tell(x_data, y_data, noise_variances = noise, append = False), GP)
    return future

def train(client, hyperparameter_bounds, GP,  max_iter = 1000, method = "mcmc"):
    future  = client.submit(lambda obj: obj.train(hyperparameter_bounds,  max_iter = max_iter, method = "mcmc"), GP)
    return future

def ask(client, candidates, GP, n, acq_func):
    future  = client.submit(lambda obj: obj.ask(input_set= candidates, n = n, acquisition_function=acq_func, vectorized = True), GP)
    return future

def log_likelihood(client, GP):
    return client.submit(lambda obj: obj.log_likelihood(), GP).result()

def get_data(client, GP):
    return client.submit(lambda obj: obj.get_data(), GP).result()

def posterior_mean(client, x_pred, GP):
    f = client.submit(lambda obj: obj.posterior_mean(x_pred), GP)
    return f

def posterior_covariance(client, x_pred, GP):
    f = client.submit(lambda obj: obj.posterior_covariance(x_pred, variance_only=True), GP)
    return f

def set_new_kernel(client, kernel, GP):
    def setk(obj):
        obj.prior.kernel=kernel
        obj.posterior.kernel=kernel
    f = client.submit(setk, GP).result()
    return f

def set_args(client, GP, args):
    f = client.submit(lambda obj: obj.set_args(args), GP).result()
    return f


def set_new_noise_func(client, noise, GP):
    def setn(obj):
        obj.likelihood.noise_function=noise
    f = client.submit(setn, GP).result()
    return f
    
######################################

def filter_xyz_data(x_data, y_data, tol):
    """
    This function takes data on a grid and transforms it into x,y, z coordinates.
    x and y are the indices, and z are the values defined on the grid.
    """
    assert tol >= 0., tol
    nonzero_index = np.where(abs(y_data) >= tol)
    return x_data[nonzero_index], y_data[nonzero_index]

def write_file(gpcam_path, chombo_path, a):
    print("Write new suggestions file:")
    date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    np.savetxt(gpcam_path+"suggestions.csv.tmp", a, delimiter=",")
    shutil.copy(gpcam_path+"suggestions.csv.tmp", gpcam_path+"suggestions."+date_time+".csv")
    os.rename(gpcam_path+'suggestions.csv.tmp', chombo_path+'suggestions.csv')

def normalize_data(vec):
    if np.all(vec) == 0.0: return vec, 0., 0.
    mi = np.min(vec)
    vec = vec - mi
    ma = np.max(vec)
    vec = vec / ma
    return vec, mi, ma

def read_file(chombo_path, filename, index, tol_ratio, delete=True, normalize = False):
    print("Waiting for Chombo file:")
    print(". ", end='')
    while True:
        if os.path.exists(chombo_path+"ready.txt") and os.path.exists(chombo_path+filename):
            while True:
                try:
                    dicct = read_hdf5(chombo_path+filename, index = index)
                    print(f"✅ Successfully read {filename}")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to read {filename}: {e}")
                    print(f"Retrying in 1 second...")
                    time.sleep(1)
                    
            if isinstance(dicct["global y"], np.ndarray): break
            else:
                print(data)
                raise Exception("Wrong data format communicated")
        else:
            print(". ", end='')
            time.sleep(0.5)
    filter_tol = tol_ratio * np.max(abs(dicct["global y"])) 
    datasets = dicct["block data"]
    dicct["global x"], dicct["global y"] = filter_xyz_data(dicct["global x"], dicct["global y"], filter_tol)
    if normalize: dicct["global y"], mi, ma = normalize_data(dicct["global y"])
    for ID in datasets: 
        comp_grid = datasets[ID][0].copy()
        x, y = filter_xyz_data(datasets[ID][0], datasets[ID][1], tol=filter_tol)
        if normalize: 
            y = y - mi
            y = y / ma
        datasets[ID] = (x,y, datasets[ID][2], comp_grid)
    if delete:
        os.remove(chombo_path+filename)
        os.remove(chombo_path+"ready.txt")
    return datasets, dicct["domain"], dicct["global x"], dicct["global y"]

def read_fileII(chombo_path, filename, index, tol_ratio, delete=True, normalize = False):
    print("Waiting for Chombo file:")
    print(". ", end='')
    while True:
        if os.path.exists(chombo_path+"ready.txt") and os.path.exists(chombo_path+filename):
            while True:
                try:
                    dicct = read_hdf5II(chombo_path+filename, index = index)
                    print(f"✅ Successfully read {filename}")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to read {filename}: {e}")
                    print(f"Retrying in 1 second...")
                    time.sleep(1)
                    
            if isinstance(dicct["global y"], np.ndarray): break
            else:
                print(data)
                raise Exception("Wrong data format communicated")
        else:
            print(". ", end='')
            time.sleep(0.5)
    filter_tol = tol_ratio * np.max(abs(dicct["global y"]))
    xpatches, ypatches = make_block_data(dicct["global x"], dicct["global y"], 64, 64, 16, 16) ###64, and 16 has to be returned from the read_file
    dicct["global x"], dicct["global y"] = filter_xyz_data(dicct["global x"], dicct["global y"], filter_tol)
    if normalize: dicct["global y"], mi, ma = normalize_data(dicct["global y"])
    for i in range(len(ypatches)):
        if ma != 0.0:
            ypatches[i] = ypatches[i] - mi
            ypatches[i] = ypatches[i] / ma
    if delete:
        os.remove(chombo_path+filename)
        os.remove(chombo_path+"ready.txt")
    return dicct["domain"], dicct["global x"], dicct["global y"], xpatches, ypatches


def scatter(x,y,z, xlim = None, ylim = None):
    fig = plt.figure(figsize=(20,5))
    norm = plt.Normalize(vmin=z.min(), vmax=z.max())
    cmap = plt.cm.viridis

    # Get RGBA colors
    colors = cmap(norm(z))

    # Set alpha=0 (transparent) where z == 0
    colors[z == 0, -1] = 0.0
    plt.scatter(x, y, c = colors)
    # Define plotting domain (x- and y-axis ranges)
    if xlim is not None: plt.xlim(xlim[0], xlim[1]) 
    if ylim is not None: plt.ylim(ylim[0], ylim[1]) 
    plt.colorbar()
    plt.show()


def plot2d(x,y,z, suggestions = None, res = 200, title = "title", filename = None):
    xi = np.linspace(x.min(), x.max(), res)
    yi = np.linspace(y.min(), y.max(), res)
    X, Y = np.meshgrid(xi, yi)
    
    # 2️⃣ Interpolate the scattered z-values onto the grid
    Z = griddata((x, y), z, (X, Y), method='cubic', fill_value = 0.)
    
    # 3️⃣ Plot as a continuous field
    plt.figure(figsize=(20,5))
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis', vmin=Z.min(), vmax=Z.max())
    plt.colorbar(label='z')
    if suggestions is not None: plt.scatter(suggestions[:,0], suggestions[:,1], s=0.1, c='black', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    if filename is not None: plt.savefig(filename)
    plt.show()

def valid(A):
    _, unique_indices = np.unique(A, axis=0, return_index=True)

    has_duplicates = len(unique_indices) != len(A)
    #print("Has duplicates:", has_duplicates)

    has_nan = np.isnan(A).any()
    has_inf = np.isinf(A).any()
    if has_duplicates or has_inf or has_nan: valid = False
    else: valid = True
    return valid

