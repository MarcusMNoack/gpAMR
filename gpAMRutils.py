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


from scipy.interpolate import griddata
from gpcam.kernels import *
from scipy.interpolate import RBFInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, identity

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
    return norm, gx, gy

def meanf(x,hps, args):
    m = args["block_mean"]
    #print(m)
    return np.zeros(len(x)) + m

def acq_func(x, gp):
    x = np.asarray(x)
    return np.sqrt(gp.posterior_covariance(x, variance_only=True)["v(x)"])


def kernelPDEII(x1, x2, hps, args):
    """
    Anisotropic non-stationary Gibbs kernel (gpAMR Eq. 4), directional extension.

    Each axis gets its own length-scale field that contracts based on the PDE
    solution's gradient COMPONENT along that axis (dq/dx_d), NOT the gradient
    norm |grad q|. Using the norm makes both axes contract identically
    (isotropic), which defeats the directional intent. The gradient sensitivity
    `beta` is shared across directions:

        ell_d(x) = ell0_d / (1 + beta * |dq/dx_d|)              d = 0 .. D-1

        k(x, x') = sf^2
                   * PROD_d [ 2 * l1_d * l2_d / (l1_d^2 + l2_d^2) ]^(1/2)   # prefactor
                   * exp( - SUM_d (x_d - x'_d)^2 / (l1_d^2 + l2_d^2) )      # anisotropic SE

    This is the diagonal Paciorek-Schervish construction. It is guaranteed
    symmetric positive semidefinite for ANY per-axis field ell_d(x) > 0, which
    the denominator (1 + beta*|g_d|) >= 1 with ell0_d > 0, beta >= 0 enforces.
    In the zero-gradient limit it reduces to a standard anisotropic SE kernel;
    setting ell0_x = ell0_y and swapping components for the norm recovers the
    original isotropic kernel.

    Hyperparameters (D = 2 case shown; layout generalizes to any D):
        hps[0] : sf       signal-std scale       -> kernel returns sf^2 * (...)
        hps[1] : ell0_x   max length scale along axis 0 (flat-region corr. length)
        hps[2] : ell0_y   max length scale along axis 1 (flat-region corr. length)
        hps[3] : b        raw gradient sensitivity; effective beta = b^2 >= 0,
                          SHARED across directions (b^2 keeps it non-negative)

    For general D: hps[1 : 1+D] are the per-axis ell0_d, hps[1+D] is b.

    Recommended training bounds  (grid/index units: domain [0,64]^2, dx ~ 1)
    ----------------------------------------------------------------------------
        hps[0] : [1e-2, 1e1]    data-dependent; ~ sqrt(mean-square of y) is the
                                center of mass. Widen if y is not O(1).
        hps[1] : [2.0, 64.0]    >=2 cells (resolution floor) .. domain extent.
        hps[2] : [2.0, 64.0]    same. Search these in LOG space if fvGP allows;
                                length scales are multiplicative.
        hps[3] : [0.0, 2.2]     effective beta in [0, ~5]. 0.0 is a legitimate
                                stationary fallback the optimizer can rest on.
                                At |grad q| ~ 1 the effective beta reads directly
                                as the contraction factor (beta=1 halves ell).

    As a ready-to-use box:
        hps_bounds = np.array([[1e-2, 1e1],
                               [2.0, 64.0],
                               [2.0, 64.0],
                               [0.0, 2.2]])

    Conditioning note
    -----------------
    On a dense grid the not-PD / Cholesky failures come from the SMOOTH regime
    (large ell0 -> long correlations -> near-duplicate rows -> rank-deficient K),
    NOT from steep/contracted regions (those are well conditioned). Add a small
    diagonal jitter ~ 1e-6 * sf^2 to K; that is what protects the large-ell0
    corner. If you later need beta > ~5 with a hard floor on ell, switch to the
    floored field  ell_d = ell_min + (ell0_d - ell_min) / (1 + beta*|g_d|).
    """
    D = x1.shape[1]
    rbf = args["rbf"]
    span = args["span"]

    # --- gradient COMPONENTS at each point ---
    # interpolator_grad returns (|grad q|, dq/dx, dq/dy). We must use the
    # SEPARATE components, NOT the magnitude: the magnitude drives both axes
    # identically (isotropic contraction). Column order here MUST match the
    # column order of x1/x2 -- i.e. x1[:, 0] is the axis gx differentiates.
    _, gx1, gy1 = interpolator_grad(rbf, span, x1)
    _, gx2, gy2 = interpolator_grad(rbf, span, x2)
    del rbf
    gc1 = np.abs(np.stack([gx1, gy1], axis=1))        # (N1, 2): [|dq/dx|, |dq/dy|]
    gc2 = np.abs(np.stack([gx2, gy2], axis=1))        # (N2, 2)

    #FLOORED VERSION
    # hps[0]=sf, hps[1]=ell0_x, hps[2]=ell0_y, hps[3]=ell_min, hps[4]=sqrt(beta)
    # ell0    = np.asarray(hps[1:1 + D], dtype=float)   # (D,) per-axis MAX length scale
    # ell_min = 0.1                              # shared floor (> 0, < min ell0_d)
    # beta    = hps[1 + D] ** 2

    # l1 = ell_min + (ell0 - ell_min)[None, :] / (1.0 + beta * gc1)   # (N1, D)
    # l2 = ell_min + (ell0 - ell_min)[None, :] / (1.0 + beta * gc2)   # (N2, D)

    
    # --- per-direction length-scale fields (shared beta, forced non-negative) ---
    ell0 = np.asarray(hps[1:1 + D], dtype=float)       # (D,) = [ell0_x, ell0_y, ...]
    beta = hps[1 + D] ** 2                             # >= 0, shared across axes
    l1 = ell0[None, :] / (1.0 + beta * gc1)            # (N1, D)  contracts per axis
    l2 = ell0[None, :] / (1.0 + beta * gc2)            # (N2, D)

    # --- diagonal Gibbs kernel: independent product over dimensions ---
    prefactor = np.ones((x1.shape[0], x2.shape[0]))
    exponent = np.zeros_like(prefactor)
    for d in range(D):
        a = l1[:, d][:, None]                          # (N1, 1)
        b = l2[:, d][None, :]                          # (1, N2)
        denom_d = a ** 2 + b ** 2                      # ell_d(x1)^2 + ell_d(x2)^2
        dx_d = x1[:, d][:, None] - x2[:, d][None, :]   # per-axis coordinate diff
        prefactor *= np.sqrt(2.0 * a * b / denom_d)    # PSD-preserving amplitude
        exponent += -(dx_d ** 2) / denom_d
    gibbs = prefactor * np.exp(exponent)

    return hps[0] ** 2 * gibbs