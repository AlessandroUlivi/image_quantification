import numpy as np
from scipy.spatial import ConvexHull


def get_convex_hull_fractions(arr_1, arr_2, roi__mask=None, threshold_arr_1=0, threshold_arr_2=0, threshold_roi_mask=0, px_thre_arr_1=3, px_thre_arr_2=3,
                              val_4_arr1_NOpassthres_arr2_passthres=0.0, val_4_arr2_NOpassthres=None):
    """
    returns the fractions of input arrays convex hulls.

    Inputs:
    - arr_1. ndarray. The array undergoes a binarization step. Pixels whose value is >threshold_arr_1 are considered pixels of interest, the rest background.
    - arr_2. ndarray of the same shape of arr_1. The array undergoes a binarization step. Pixels whose value is >threshold_arr_2 are considered pixels of interest, the rest background.
    - roi__mask. Optional. ndarray of the same shape of arr_1. If provided, restricts the analysis a region of interest identified by pixels whose value is >threshold_roi_mask.
    - threshold_arr_1. int or float. Default 0. Defines the highpass threshold to distinguish pixels of interest from background pixels in arr_1.
    - threshold_arr_2. int or float. Default 0. Defines the highpass threshold to distinguish pixels of interest from background pixels in arr_2.
    - threshold_roi_mask. int or float. Default 0. Only applies if roi__mask is provided. Defines the highpass threshold to distinguish pixels of interest from background pixels
    in roi__mask.
    - px_thre_arr_1. int or float. Default 3. It must be >=3. The minimum number of pixels of interest in arr_1 for the measurement to be made.
    - px_thre_arr_2. int or float. Default 3. It must be >=3. The minimum number of pixels of interest in arr_1 for the measurement to be made.
    - val_4_arr1_NOpassthres_arr2_passthres. any. Default 0.0. The value to return if the number of pixels of interest in arr_2 >px_thre_arr_2 and the number of pixels
    of interest in arr_1 <px_thre_arr_1.
    -  val_4_arr2_NOpassthres. any. Default None. The value to return if the number of pixels of interest in arr_2 <px_thre_arr_2, irrespective of the number of pixels
    of interest in arr_1.

    Outputs:
    - if the number of pixels of interest in arr_2 >px_thre_arr_2 and the number of pixels of interest in arr_1 >px_thre_arr_2.
    float. Area of the convex hull calculated on arr_1 pixels of interest / Area of the convex hull calculated on arr_2 pixels of interest.
    - if the number of pixels of interest in arr_2 >px_thre_arr_2 and the number of pixels of interest in arr_1 <px_thre_arr_2. val_4_arr1_NOpassthres_arr2_passthres is returned.
    Default 0.0.
    - if the number of pixels of interest in arr_2 <px_thre_arr_2 irrespective of the number of pixels of interest in arr_1. val_4_arr2_NOpassthres is returned. Default None.
    """
    assert px_thre_arr_1>=3, 'px_thre_arr_1 must be >=3 to calculate a convex hull'
    assert px_thre_arr_2>=3, 'px_thre_arr_2 must be >=3 to calculate a convex hull'

    #Copy input arrays, threshold them using their respective threshold values, and set the values to 0 and 1
    arr_1_copy = np.where(arr_1>threshold_arr_1, 1, 0)
    arr_2_copy = np.where(arr_2>threshold_arr_2, 1, 0)

    #If roi_mask is provided, copy it, threshold it threshold_roi_mask, set its values to 0 and 1,
    # and use it to set to 0 pixels in arr_1 and arr_2 which are corresponded by background pixels in roi_mask
    if hasattr(roi__mask, "__len__"):
        roi__mask_01 = np.where(roi__mask>threshold_roi_mask, 1,0)
        arr_1_to_proc = np.where(roi__mask_01>0, arr_1_copy, 0)
        arr_2_to_proc = np.where(roi__mask_01>0, arr_2_copy, 0)
    else:
        arr_1_to_proc = arr_1_copy
        arr_2_to_proc = arr_2_copy
    
    #Count the number of pixels of interest in arr_1_to_proc and arr_2_to_proc
    poi_arr_1 = np.sum(arr_1_to_proc)
    poi_arr_2 = np.sum(arr_2_to_proc)

    #Return val_4_arr2_NOpassthres if poi_arr_2<px_thre_arr_2
    if poi_arr_2<px_thre_arr_2:
        return val_4_arr2_NOpassthres
    
    else:
        #Return val_4_arr1_NOpassthres_arr2_passthres if poi_arr_1<px_thre_arr_1
        if poi_arr_1<px_thre_arr_1:
            return val_4_arr1_NOpassthres_arr2_passthres
        
        else:
            #Get the coordinates of pixels of interest in arr_1_to_proc and arr_2_to_proc
            arr_1_pixels_coords = np.argwhere(arr_1_to_proc>0)
            arr_2_pixels_coords = np.argwhere(arr_2_to_proc>0)

            #Get the convex hulls for arr_1 and arr_2
            arr_1_convexhull = ConvexHull(arr_1_pixels_coords)
            arr_2_convexhull = ConvexHull(arr_2_pixels_coords)

            #Get the volumes of arr_1 and arr_2 convex hulls - NOTE: when arr_1 and arr_2 are 2D arrays, the area is returned
            arr_1_vol = arr_1_convexhull.volume
            arr_2_vol = arr_2_convexhull.volume

            return arr_1_vol/arr_2_vol
