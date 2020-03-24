import os
import time
import psutil
import multiprocessing
import numpy as np
import cv2 as cv
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.colors as col

import math
from pylab import *
from scipy import interpolate
from scipy.stats import kurtosis
from scipy.linalg import circulant
from scipy.signal import find_peaks
from scipy.interpolate import splev, splrep, BSpline
from scipy.ndimage.filters import gaussian_filter


def print_mem():
    info = psutil.virtual_memory()
    print(u'内存使用：',psutil.Process(os.getpid()).memory_info().rss)
    print(u'总内存：',info.total)
    print('Memory usage：'+str(info.percent)+'%')
    print(u'cpu个数：',psutil.cpu_count()) 
    

def multi_proc(func, args, n_core):
    start = time.time()
    p = multiprocessing.Pool(n_core)
    out = p.map_async(func, args).get()
    p.close()
    p.join()
    tcost = time.time() - start
    print('%f seconds'%tcost)
    return out

def div0( a, b ):
    '''
    ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] 
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def find_shape(series_file_names):
    '''
    return the average time interval between scans and the number of slices, called by the function read_dcm,
    '''

    origins, c = [], []
    for x in series_file_names:
        d = sitk.ReadImage(x).GetOrigin()[2]
        t = float(sitk.ReadImage(x).GetMetaData(key='0008|0032'))
        if not d in origins:
            origins.append(d)
            c.append([t])
        else:
            idx = origins.index(d)
            c[idx].append(t)
    n_slices = len(origins)

    avgs = []
    for i in range(n_slices):
        c[i] = sorted(c[i])
        for j in range(1,len(c[i])):
            c[i][j-1] =round(c[i][j] - c[i][j-1], 2)
        c[i] = c[i][0:-1]
        c[i].remove(max(c[i]))
        avg = (round(sum(c[i])/len(c[i]), 2))
        avgs.append(avg)

    dt = round(sum(avgs)/len(avgs), 2)


    return dt, n_slices

def read_dcm(path, id):
    '''
    read dicom files according to series_ids. 
    return array, time interval, number of slices and tags
    '''
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, id, recursive=False, useSeriesDetails=False)
    files=list(files)
    files.sort(key=lambda x:float(sitk.ReadImage(x).GetMetaData(key='0020|0013')))     
    dt, n_slices = find_shape(files)    

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)       
    
    try: 
        img = reader.Execute()
    except:
        print('error in id: %s \n'%id)

    tags = []
    tags.append(img.GetOrigin())
    tags.append(img.GetSpacing())
    tags.append(img.GetDirection())
    tags.append(sitk.ReadImage(files[0]).GetMetaData(key='0018|0081')) # Echo Time

    arr = sitk.GetArrayFromImage(img)

    return arr, dt, n_slices, tags


def interp_bsplime(args):
    [sig, times, times_new, mask] = args
    # if mask == 0: return np.zeros(len(times_new), dtype=np.int16)
    mask = np.array([mask]*sig.shape[0])
    sig  = sig*mask
    t, c, k = splrep(times, sig, s = 0)
    spline = BSpline(t, c, k)
    sig_interp = spline(times_new)
    return sig_interp

def interpolation(arr, mask, dt, n_slices):
    '''
    interpolating the original signal from dt seconds to 1 seconds
    return an array of interpolated signals
    '''

    count = round(arr.shape[0]/n_slices)
    total_t=dt*count
    times = np.linspace(0, total_t, count, endpoint=False)
    times_new=np.linspace(0, total_t, total_t, endpoint=False)
    arr_interp = np.zeros((count, n_slices, arr.shape[1], arr.shape[2]))
    print(arr_interp.shape)

    for t in range(count):
        try:
            arr_interp[t,:,:,:] = arr[(t*n_slices):((t+1)*n_slices), :, :]
            # mask = np.array([mask]*arr_interp.shape[0])
            # arr_interp  = arr_interp*mask
        except:
            print('error in id: %s \n'%series_id)
            continue

    [nz, nx, ny] = mask.shape
    
    # run interpolation with multiprocessing
    args = [(arr_interp[:,z,x,y], times, times_new, mask[z,x,y]) for z in range(nz) for x in range(nx) for y in range(ny)]
    arr_interp = multi_proc(interp_bsplime, args, 10)
    print(np.min(arr_interp), np.max(arr_interp))
    # convert result to array
    print('Done, converting result to array...')
    nt = len(times_new)
    arr_interp = np.array(arr_interp).reshape(nz,nx,ny,nt) 
    arr_interp = np.moveaxis(arr_interp, -1, 0)
    [nt,nz,nx,ny] = arr_interp.shape
    return arr_interp


'''

# find mask of bone and area outside of bone
def get_mask_brain(arr, loth, hith):
    [nz, nx, ny] = arr.shape
    tmp0 = np.where(arr<loth, 0, 1).astype(np.int16)
    tmp1 = np.where(arr>hith, 0, 1).astype(np.int16)
    tmp2 = tmp1
    for z in range (nz):
        for x in range(nx):
            L,R = ny//2, ny//2
            idx = np.where(tmp1[z,x,:]==0)[0]
            if len(idx)>0:
                L=idx[0]
                R=idx[-1]
            tmp1[z,x,0:L] = 0
            tmp1[z,x,R:-1] = 0
        for y in range(ny):
            B,U = nx//2, nx//2
            idx = np.where(tmp2[z,:,y]==0)[0]
            if len(idx)>0:
                U=idx[0]
                B=idx[-1]
            tmp2[z,0:U,y] = 0
            tmp2[z,B:-1,y] = 0
    return tmp0*tmp1*tmp2

# find mask of bone and bone only
def get_mask_bone(arr, loth, hith):
    arr[arr>hith]=0
    return np.where(arr>loth, 1, 0)

'''
def gausskernel(size):
    """1d gaussian kernel
    """
    sigma=1.0
    gausskernel=np.zeros(size,np.float32)
    for i in range (size):
        norm=math.pow(i-(size-1)/2, 2)
        gausskernel[i]=math.exp(-norm/(2*math.pow(sigma,2)))   # 求高斯卷积
    sum=np.sum(gausskernel)   # 求和
    kernel=gausskernel/sum   # 归一化
    return kernel
def der1(sig):
    """return the temporal derivative of a time signal
    """
    l = len(sig)
    out = np.zeros(l, np.float32)
    for i in range(1,l-1):
        out[i] = (sig[i+1]-sig[i-1])/2
    out[0] = sig[1]- sig[0]
    out[-1] = sig[-1]- sig[-2]
    return out

def gauss_der(der, k):
    """return the gaussian derivative of a time signal
    """
    l = len(der)
    w = len(k)
    wh = (w-1)//2
    out = np.copy(der)
    for i in range(wh, l-1-wh):
        out[i] = np.dot(der[i-wh:i+wh+1], k)
    return out

    
def total_change(args):
    [sig, mask, kernel] = args
    if mask == 0: return 0
    if np.max(sig) < 20: return 0
    
    if not 5 < np.argmax(sig): return 0
    peaks, properties = find_peaks(sig, height=35, prominence=15, width=5, rel_height=1)
    if len(peaks) != 1: return 0

    #kur = kurtosis(sig, fisher=True)
    #if not -1 < kur < 3 : return 0
    der = der1(sig)
    gder = gauss_der(der, kernel)
    return 0.5*np.sum(abs(gder[0:-2]) + abs(gder[1:-1]))


def get_invAIF(aif, lamb):
    l = len(aif)
    print(l)
    aif = np.hstack((aif, np.zeros(l)))
    aif = circulant(aif)
    U,s,V = np.linalg.svd(aif)
    th = np.max(s)*lamb
    S = np.diag(s)
    invS = np.linalg.pinv(S)
    truncation_mask = np.where(S<th, 0, 1)
    tinvS = invS*truncation_mask
    invAIF = (V.T).dot(tinvS).dot(U.T)
    return invAIF


def ctSVD(arg):
    [invAIF, tac] = arg
    l = len(tac)
    tac = np.hstack((tac, np.zeros(l)))
    return invAIF.dot(tac)[0:l]


def plot_maps(arr, slice_i, plot_path, CBF, CBV, MTT, Tmax, penumbra):
    # plot feature maps and penumbra
    fig, ax = plt.subplots(2,3,figsize=(25,10))
    cbar = ['blue','cyan','green','orange','red']

    ct = col.LinearSegmentedColormap.from_list('ct',cbar)
    cbar.reverse()
    ct_r = col.LinearSegmentedColormap.from_list('ct_r',cbar)
    cm.register_cmap(cmap=ct)
    cm.register_cmap(cmap=ct_r)

    cax = ax[0,0].imshow(arr[0,slice_i,:,:], cmap='gray')
    fig.colorbar(cax, ax=ax[0,0])
    ax[0,0].set_title('Base')
    ax[0,0].axis('off')

    cax = ax[0,1].imshow(CBF[slice_i,:,:], cmap='jet')
    fig.colorbar(cax, ax=ax[0,1])
    ax[0,1].set_title('CBF')
    ax[0,1].axis('off')

    cax = ax[0,2].imshow(CBV[slice_i,:,:], cmap='jet')
    fig.colorbar(cax, ax=ax[0,2])
    ax[0,2].set_title('CBV')
    ax[0,2].axis('off')

    cax = ax[1,0].imshow(MTT[slice_i,:,:], cmap='jet_r')
    fig.colorbar(cax, ax=ax[1,0])
    ax[1,0].set_title('MTT')
    ax[1,0].axis('off')

    cax = ax[1,1].imshow(Tmax[slice_i,:,:], cmap='jet_r') 
    fig.colorbar(cax, ax=ax[1,1])
    ax[1,1].set_title('Tmax')
    ax[1,1].axis('off')

    cax = ax[1,2].imshow(penumbra[slice_i,:,:], cmap='jet_r') 
    fig.colorbar(cax, ax=ax[1,2])
    ax[1,2].set_title('Penumbra')
    ax[1,2].axis('off')

    fig.savefig(plot_path+str(slice_i)+".png")
    plt.show()
    plt.close()    

    

#         def MyMedianAverage(inputs,width):
#             w = width//2
#             inputs_new = np.concatenate(([inputs[0]]*w, inputs, [inputs[len(inputs)-1]]*w ), axis=0)
#             for i in range(w,len(inputs)-w):
#                 buff = inputs_new[i-w:i+w]
#                 buff = np.delete(buff,np.argmax(buff))
#                 buff = np.delete(buff,np.argmin(buff))
#                 inputs[i] = np.mean(buff)
#             return inputs
