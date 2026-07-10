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
            except Exception as e:
                print("ERROR: ", e)
                continue
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

def set_hps(client, GP, hps):
    f = client.submit(lambda obj: obj.set_hyperparameters(hps), GP).result()
    return f

def set_new_noise_func(client, noise, GP):
    def setn(obj):
        obj.likelihood.noise_function=noise
    f = client.submit(setn, GP).result()
    return f
    
######################################

def filter_all_data(data, tol_ratio):
    filter_tol = tol_ratio * np.max(abs(data["global_coordinates"]))
    for blockID in data["local_data"]:
        y_data = data["local_data"][blockID]["values"]
        x_data = data["local_data"][blockID]["points"]
        data["local_data"][blockID]["points"], data["local_data"][blockID]["values"] = filter(x_data, y_data, filter_tol)
    return data

def filter(x_data, y_data, tol):
    """
    This function takes data on a grid and transforms it into x,y, z coordinates.
    x and y are the indices, and z are the values defined on the grid.
    """
    
    assert tol >= 0., tol
    nonzero_index = np.where(abs(y_data) >= tol)
    if np.all(abs(y_data) < tol): return x_data, y_data
    return x_data[nonzero_index], y_data[nonzero_index]

def normalize_data(data, mi = None, ma = None):
    v = data["global_funcvalues"]
    vmin, vmax = v.min(), v.max()
    for blockID in data["local_data"]:
        y_data = data["local_data"][blockID]["values"]
        data["local_data"][blockID]["values"] = normalize(y_data, vmin, vmax)
    return data

def normalize(vec, vmin, vmax):
    if vmin == vmax == 0.: return vec #only when vec.all() == 0., return all 0s
    elif vmin == vmax: return vec/vmax #only when min(vec) == max(vec), return vector of ones
    return (vec - vmin) / (vmax - vmin)


def write_file(gpcam_path, chombo_path, a):
    print("Write new suggestions file:")
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    np.savetxt(gpcam_path+"suggestions.csv.tmp", a, delimiter=",")
    shutil.copy(gpcam_path+"suggestions.csv.tmp", gpcam_path+"suggestions."+date_time+".csv")
    os.rename(gpcam_path+'suggestions.csv.tmp', chombo_path+'suggestions.csv')

def refined_grid(refinement, x_range, y_range):
    (x0, x1), (y0, y1) = x_range, y_range
    nx = round((x1 - x0) * refinement) + 1
    ny = round((y1 - y0) * refinement) + 1
    xs = np.linspace(x0, x1, nx)   # includes both endpoints
    ys = np.linspace(y0, y1, ny)
    return np.stack(np.meshgrid(xs, ys, indexing='ij'), axis=-1).reshape(-1, 2)



# def read_file(chombo_path, filename, index, tol_ratio, delete=True, normalize = False, filter = True):
#     print("Waiting for Chombo file:")
#     print(". ", end='')
#     while True:
#         if os.path.exists(chombo_path+"ready.txt") and os.path.exists(chombo_path+filename):
#             while True:
#                 try:
#                     dicct = read_hdf5(chombo_path+filename, index = index)
#                     print(f"✅ Successfully read {filename}")
#                     break
#                 except Exception as e:
#                     print(f"⚠️  Failed to read {filename}: {e}")
#                     print(f"Retrying in 1 second...")
#                     time.sleep(1)
                    
#             if isinstance(dicct["global y"], np.ndarray): break
#             else:
#                 print(data)
#                 raise Exception("Wrong data format communicated")
#         else:
#             print(". ", end='')
#             time.sleep(0.5)
#     filter_tol = tol_ratio * np.max(abs(dicct["global y"])) 
#     datasets = dicct["block data"]
#     dicct["global x"], dicct["global y"] = filter_xyz_data(dicct["global x"], dicct["global y"], filter_tol, filter = filter)
#     if normalize: dicct["global y"], mi, ma = normalize_data(dicct["global y"])
#     for ID in datasets: 
#         comp_grid = datasets[ID][0].copy()
#         x, y = filter_xyz_data(datasets[ID][0], datasets[ID][1], tol=filter_tol, filter = filter)
#         if normalize: 
#             y = y - mi
#             y = y / ma
#         #datasets[ID] = (x,y, datasets[ID][2], comp_grid)
#         datasets[ID] = (x,y, datasets[ID][2])
#     if delete:
#         date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         os.rename(chombo_path+filename, chombo_path+filename+date_time)
#         #os.remove(chombo_path+filename)
#         os.remove(chombo_path+"ready.txt")
#     return datasets, dicct["domain"], dicct["global x"], dicct["global y"]

# def read_fileII(chombo_path, filename, index, tol_ratio, delete=True, normalize = False):
#     print("Waiting for Chombo file:")
#     print(". ", end='')
#     while True:
#         if os.path.exists(chombo_path+"ready.txt") and os.path.exists(chombo_path+filename):
#             while True:
#                 try:
#                     dicct = read_hdf5II(chombo_path+filename, index = index)
#                     print(f"✅ Successfully read {filename}")
#                     break
#                 except Exception as e:
#                     print(f"⚠️  Failed to read {filename}: {e}")
#                     print(f"Retrying in 1 second...")
#                     time.sleep(1)
                    
#             if isinstance(dicct["global y"], np.ndarray): break
#             else:
#                 print(data)
#                 raise Exception("Wrong data format communicated")
#         else:
#             print(". ", end='')
#             time.sleep(0.5)
#     filter_tol = tol_ratio * np.max(abs(dicct["global y"]))
#     xpatches, ypatches = make_block_data(dicct["global x"], dicct["global y"], 64, 64, 16, 16) ###64, and 16 has to be returned from the read_file
#     dicct["global x"], dicct["global y"] = filter_xyz_data(dicct["global x"], dicct["global y"], filter_tol, filter = filter)
#     if normalize: dicct["global y"], mi, ma = normalize_data(dicct["global y"])
#     for i in range(len(ypatches)):
#         if ma != 0.0:
#             ypatches[i] = ypatches[i] - mi
#             ypatches[i] = ypatches[i] / ma
#     if delete:
#         date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#         os.rename(chombo_path+filename, chombo_path+filename+date_time)
#         #os.remove(chombo_path+filename)
#         os.remove(chombo_path+"ready.txt")
#     return dicct["domain"], dicct["global x"], dicct["global y"], xpatches, ypatches



def read_fileIII(chombo_path, filename, index, n_sub_x=1, n_sub_y=1, pad_x=0, pad_y=0, rename=True, delete=False):
    print("Waiting for Chombo file:", chombo_path+filename)
    print(". ", end='')
    while True:
        if os.path.exists(chombo_path+"ready.txt") and os.path.exists(chombo_path+filename):
            while True:
                try:
                    dicct = read_hdf5_decomposed(chombo_path+filename, index=index,
                             n_sub_x=n_sub_x, n_sub_y=n_sub_y, pad_x=pad_x, pad_y=pad_y)
                    print(f"✅ Successfully read {filename}")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to read {filename}: {e}")
                    print(f"Retrying in 1 second...")
                    time.sleep(1)
                    
            if isinstance(dicct["global_coordinates"], np.ndarray): break
            else:
                print(data)
                raise Exception("Wrong data format communicated")
        else:
            print(". ", end='')
            time.sleep(0.5)

    if delete:
        os.remove(chombo_path+filename)
        os.remove(chombo_path+"ready.txt")
    elif rename:
        date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        os.rename(chombo_path+filename, chombo_path+filename+date_time)
        #os.remove(chombo_path+filename)
        os.remove(chombo_path+"ready.txt")
    else:
        pass
    
    return dicct


def make_refinement_candidates(result, refinement_level, include_coarse=False):
    """
    Generate refinement candidate points for every subdomain.
 
    refinement_level in {2, 4, 8}: number of cells each coarse cell is split
    into per axis. 2 -> one new point between neighbors, 4 -> three, 8 -> seven
    (fine spacing h = 1 / refinement_level on both axes).
 
    Ownership / uniqueness: each fine point's owning coarse cell is
    floor(fine_index / refinement_level); since the subdomain `interior` ranges
    tile the coarse domain exactly, every fine point maps to exactly one
    subdomain. Local domains overlap through their halos, but the candidate sets
    do not -- their union is the complete fine grid over the global domain with
    no shared points. By default the original coarse points (which already hold
    data) are excluded, so only genuinely new locations are returned.
 
    Parameters
    ----------
    result : dict from read_hdf5_decomposed (uses "domain" and "local_data").
    refinement_level : int in {2, 4, 8}.
    include_coarse : if True, keep the existing coarse points too (rarely wanted).
 
    Returns
    -------
    dict with keys:
      "local_candidates" : {id: (Ki, 2)} new points owned by each subdomain
      "global_candidates": (K, 2) concatenation of all of the above (unique)
      "refinement_level" : the level used
    """
    if refinement_level not in (2, 4, 8):
        raise ValueError(f"refinement_level must be 2, 4 or 8, got {refinement_level}")
    r = refinement_level
 
    (x0, x1), (y0, y1) = result["domain"]            # [[x0,x1],[y0,y1]]
    nx = int(round(x1 - x0)) + 1                      # coarse points along x
    ny = int(round(y1 - y0)) + 1
    nfx = (nx - 1) * r + 1                            # fine points along x
    nfy = (ny - 1) * r + 1
    h = 1.0 / r
 
    local_candidates = {}
    for sid, sub in result["local_data"].items():
        c0, c1 = sub["bounds"]["interior_cols"]      # owned coarse cols [c0, c1)
        r0, r1 = sub["bounds"]["interior_rows"]      # owned coarse rows [r0, r1)
 
        # fine indices owned by this subdomain (cap the last block at the edge)
        ifx = np.arange(c0 * r, min(c1 * r, nfx))
        ify = np.arange(r0 * r, min(r1 * r, nfy))
 
        IFX, IFY = np.meshgrid(ifx, ify)
        IFX, IFY = IFX.ravel(), IFY.ravel()
 
        if not include_coarse:                       # drop pure coarse points
            keep = ~((IFX % r == 0) & (IFY % r == 0))
            IFX, IFY = IFX[keep], IFY[keep]
 
        local_candidates[sid] = np.column_stack([x0 + IFX * h, y0 + IFY * h])
 
    global_candidates = (np.vstack(list(local_candidates.values()))
                         if local_candidates else np.empty((0, 2)))
 
    # by construction this should be exact; assert so misuse fails loudly
    expected = nfx * nfy - (0 if include_coarse else nx * ny)
    assert len(global_candidates) == expected, (
        f"candidate count {len(global_candidates)} != expected {expected}; "
        "interior tiles may not cover the domain")
    assert len(np.unique(global_candidates, axis=0)) == len(global_candidates), \
        "duplicate candidates across subdomains"
 
    return {
        "local_candidates":  local_candidates,
        "global_candidates": global_candidates,
        "refinement_level":  r,
    }


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


from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator
def padded_ct(points, values, pad_frac=0.1, n_ghost=60):
    lo, hi = points.min(0), points.max(0)
    span = hi - lo
    plo, phi = lo - pad_frac*span, hi + pad_frac*span
    t = np.linspace(0, 1, n_ghost)
    ring = np.vstack([
        np.c_[plo[0] + t*(phi[0]-plo[0]), np.full_like(t, plo[1])],
        np.c_[plo[0] + t*(phi[0]-plo[0]), np.full_like(t, phi[1])],
        np.c_[np.full_like(t, plo[0]), plo[1] + t*(phi[1]-plo[1])],
        np.c_[np.full_like(t, phi[0]), plo[1] + t*(phi[1]-plo[1])],
    ])
    ghost = NearestNDInterpolator(points, values)(ring)   # Neumann-style extension
    P = np.vstack([points, ring])
    V = np.concatenate([values, ghost])
    return CloughTocher2DInterpolator(P, V, fill_value=0.), span

def int_obj(x_data, y_data):
    rbf, span = padded_ct(x_data, y_data)
    return  rbf, span

def interpolator(rbf, x):
    return rbf(x)

def interpolator_grad(rbf, span, x):
    h = 1e-3 * span.min()
    ex, ey = np.array([h, 0]), np.array([0, h])
    gx = (rbf(x + ex) - rbf(x - ex)) / (2*h)
    gy = (rbf(x + ey) - rbf(x - ey)) / (2*h)
    norm = np.hypot(gx, gy)
    return norm
