import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import label, regionprops
from utils import get_euclidean_distances


def get_convex_hull_fraction(arr_1, arr_2, roi__mask_1=None, roi__mask_2=None, threshold_arr_1=0, threshold_arr_2=0, threshold_roi_mask_1=0, threshold_roi_mask_2=0, px_thre_arr_1=3, px_thre_arr_2=3,
                              val_4_arr1_NOpassthres_arr2_passthres=0.0, val_4_arr2_NOpassthres=None):
    """
    returns the fractions of input arrays convex hulls.

    Inputs:
    - arr_1. ndarray. The array undergoes a binarization step. Pixels whose value is >threshold_arr_1 are considered pixels of interest, the rest background.
    - arr_2. ndarray of the same shape of arr_1. The array undergoes a binarization step. Pixels whose value is >threshold_arr_2 are considered pixels of interest, the rest background.
    - roi__mask_1. Optional. ndarray of the same shape of arr_1. If provided, restricts the analysis of arr_1 to a region of interest identified by pixels whose value are
      >threshold_roi_mask_1.
    - roi__mask_2. Optional. ndarray of the same shape of arr_1. If provided, restricts the analysis of arr_2 to a region of interest identified by pixels whose value are
      >threshold_roi_mask_2.
    - threshold_arr_1. int or float. Default 0. Defines the highpass threshold to distinguish pixels of interest from background pixels in arr_1.
    - threshold_arr_2. int or float. Default 0. Defines the highpass threshold to distinguish pixels of interest from background pixels in arr_2.
    - threshold_roi_mask_1. int or float. Default 0. Only applies if roi__mask_1 is provided. Defines the highpass threshold to distinguish pixels of interest from background pixels
    in roi__mask_1.
    - threshold_roi_mask_2. int or float. Default 0. Only applies if roi__mask_2 is provided. Defines the highpass threshold to distinguish pixels of interest from background pixels
    in roi__mask_2.
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

    #If roi_mask_1 is provided, copy it, threshold it using threshold_roi_mask_1, set its values to 0 and 1,
    # and use it to set to 0 pixels in arr_1 which are corresponded by background pixels in roi_mask_1
    if hasattr(roi__mask_1, "__len__"):
        roi__mask_1_01 = np.where(roi__mask_1>threshold_roi_mask_1, 1,0)
        arr_1_to_proc = np.where(roi__mask_1_01>0, arr_1_copy, 0)
    else:
        arr_1_to_proc = arr_1_copy
    
    #If roi_mask_2 is provided, copy it, threshold it using threshold_roi_mask_2, set its values to 0 and 1,
    # and use it to set to 0 pixels in arr_2 which are corresponded by background pixels in roi_mask_2
    if hasattr(roi__mask_2, "__len__"):
        roi__mask_2_01 = np.where(roi__mask_2>threshold_roi_mask_2, 1,0)
        arr_2_to_proc = np.where(roi__mask_2_01>0, arr_2_copy, 0)
    else:
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


def measure_regions_euclidean_distances_within_array(label_img, roi__mask=None, desired__distance='min', transform_to_label_img=False, label_img_thres=0, return_excluded_distances=False):

    if not transform_to_label_img:
        assert np.min(label_img)==0, 'label_img must have background values set to 0 if a label image is provided'
        assert np.max(label_img)>0, 'label_img must have label region values >0 if a label image is provided'

    #Copy the input image
    label_img_copy = label_img.copy()

    #Transform input images to label images if transform_to_label_img is set to True
    if transform_to_label_img:
        #Theshold the input image using label_img_thres and set it to values 1 and 0, where 1s are assumed to be the pixels of interest
        label_img_copy_01 = np.where(label_img_copy>label_img_thres, 1,0)
        img_i = label(label_img_copy_01)
    else:
        img_i = label_img_copy
    
    #Set values outside roi to 0 (background) if roi__mask is provided
    if hasattr(roi__mask, "__len__"):
        roi__mask_copy = roi__mask.copy()
        img = np.where(roi__mask_copy>0, img_i, 0)
    else:
        img = img_i
    
    #get regionproperties of img
    regprops_img = regionprops(img)

    #get the coordinates of all the regions in img - transform them in a list of tuples
    all_regions_coords = [re.coords for re in regprops_img]

    #Initialize an output list
    output_list = []

    #If desired__distance is min or max, initialize an output dictionary, to track coordinate pairs already included in the analysis.
    # This will also be output if desired__distance=='min' or desired__distance=='max'
    #Also initialize a dictionary to double check, in case, the excluded coordinates
    if desired__distance=='min' or desired__distance=='max':
        output_dictionary_coords_dist = {}
        excluded_coordinates_dist_dict = {}

    #Initialize a dictionary to track coordinate pairs already included in the analysis - this will be output if desired__distance=='min' or desired__distance=='max'
    output_dictionary_coords_dist = {}

    #Iterate through the coordinated of the regions of img:
    for pos, r in enumerate(all_regions_coords):

        #Remove r coordinates from the general list
        all_regions_coords_copy = all_regions_coords.copy()
        all_regions_coords_copy.pop(pos)

        #All the coordinates other than r in a unique list
        pooled_all_reg_all_regions_coords = []
        for r2 in all_regions_coords_copy:
            pooled_all_reg_all_regions_coords = pooled_all_reg_all_regions_coords + list(r2)
        #Re-transform pooled coordinates in a numpy array
        pooled_all_reg_coords_arr = np.asarray(pooled_all_reg_all_regions_coords)

        #Get the min/max/mean distance of r to pixels of the rest of regions within the input image, and the coordinates of the pixel pair
        wanted_distance, pixels_coords = get_euclidean_distances(r, pooled_all_reg_coords_arr, desired_distance=desired__distance)

        #If the min or max distances are calculated, avoid duplicating pixels pairs
        if desired__distance=='min' or desired__distance=='max':
            if ((tuple(pixels_coords[0]), tuple(pixels_coords[1])) in output_dictionary_coords_dist) or ((tuple(pixels_coords[0]), tuple(pixels_coords[1])) in output_dictionary_coords_dist):
                excluded_coordinates_dist_dict[(tuple(pixels_coords[0]), tuple(pixels_coords[1]))]=wanted_distance
            else:
                output_dictionary_coords_dist[(tuple(pixels_coords[0]), tuple(pixels_coords[1]))]=wanted_distance
                output_list.append(wanted_distance)
        else:
            output_list.append(wanted_distance)

    #If desired__distance is min or max, return distances list and dictionary of paired pixels coordinates. Return the distances list and None if  desired__distance is mean
    if desired__distance=='min' or desired__distance=='max':
        if return_excluded_distances:
            return output_list, output_dictionary_coords_dist, excluded_coordinates_dist_dict
        else:
            return output_list, output_dictionary_coords_dist
    else:
        return output_list, None

