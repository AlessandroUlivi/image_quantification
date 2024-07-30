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


def measure_regions_euclidean_distances_within_array(label_img, roi__mask=None, desired__distance='min', highpass_n_regions=1, transform_to_label_img=False,
                                                     label_img_thres=0, return_excluded_distances=False, val_n_regions_nopass=None):
    """
    Returns the eucledean distances between each region of an input image (label_img) and the rest of the regions of the input image.
    NOTE: when returning the minimum or maxima euclidean distances the following scenario could happen: let's assume that the distance-ij is the minimum distance
    between region-i and the rest of the regions of label_img, and it connect with region-j. It is possible that the minimum distance between
    region-j and the rest of the regions of label_img is also distance-ij, as region-i is the closest region to region-j. This means that if calculating
    one distance per region, and including them all in the output, some distances could be duplicated. This is avoided, so each distance is only
    present once in the output list.

    Inputs:
    - label_img. ndarray. It can either be a label image (an image where pixels of separate regions are assigned the same value and a unique value is assigned to each separate region)
    or a binary image. If a label image is provided, the values must be >=0 and pixels of value 0 are assumed to correspond to the background. If a binary image is given,
    the parameter transform_to_label_img must be set to True in order for the function to transfom it in a label image using skimage.measure.label method
    (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label). If a binary mask is provided it will be firstly rescaled in the 0-1 range using the
    label_img_thres parameter. Pixels with intensity values >label_img_thres will be set to 1 and considered pixels of interest, while the rest will be
    set to 0 and considered background. The default value for label_img_thres is 0. NOTE: a binary mask can be considered a label image if only a single
    region is present.
    - roi__mask. Optional parameter. ndarray of the same shape of label_img. Binary mask. When provided, the analysis of label_img will be restricted to the region of
    interest indicated by roi__mask. It is assumed that the region of interest corresponds to pixels in roi__mask with value>0.
    - desired__distance. String. Three choices are possible: 'min' (default), 'max', 'mean'. The parameter is passed to the function get_euclidean_distances (within utils.py). If 'min'
    the minimum distances between each region of label_img and the rest of the regions of label_img are measured. If 'max' the maximum distances between each region of label_img and
    the rest of the regions of label_img are measured. If 'mean' the mean distances between each region of label_img and the rest of the regions of label_img are measured.
    - highpass_n_regions. int or float. Must be >=1. Default 1. The highpass filter for the minimum number of regions in label_img which have to be present for doint the analysis.
    - transform_to_label_img. Bool (defaul False). Specifies if label_img should be transformed to a label image. It must be True if label_img is not a label image.
    - label_img_thres. Int or float. Only applies when transform_to_label_img is True. Default 0. Defines the highpass threshold to distinguish pixels of interest from background
    in label_img. Pixels whose value is >label_img_thres are considered pixels of interest. The rest of the pixels are considered background.
    - return_excluded_distances. Bool. Default False. Only applies when desired__distance is 'min' or 'max'. When desired__distance is 'min' or 'max', distances could be duplicated
    as the min/max distance between region-i and region-j can be the same min/max distance between region-j and region-i. This duplication is avoided by default. If
    return_excluded_distances, coordinate pairs which have been excluded from the ouput because duplicatded, are also returned.
    - val_n_regions_nopass. The value which is returned if n_region highpass filter is not passed. Default None.

    Outputs:
    I will use i to refer to the i-th region of label_img and j to refer to a separate j-th region of label_img. n is the total number of separate regions in label_img.
    - there are less or equal to highpass_n_regions number of regions in label_img. A tuple is returned with val_n_regions_nopass value both in position 0 and in position 1.
    - if there are more than highpass_n_regions number of regions in label_img.
        - if desired__distance=='min'.
            - if return_excluded_distances==False. the output is a tuple.
                Position-0 is a list collecting, per each i region, the minimum euclidean distance with the rest of regions of
                label_img, but avoiding duplication of distances when the same distance-ij is the minimum distance both between region-i
                and the rest of the regions of label_img and of region-j and the rest of the regions of label_img.
                Position-1 is a dictionary. Per each distance of the output at position-0, the coordinates of the pixel pair for which the
                distance has been calculated correspond to the key of the dictionary and the are linked to their distance as the dictionary value.
            - if return_excluded_distances==True. the output is a tuple.
                = Position-0 is a list collecting, per each i region, the minimum euclidean distance with the rest of regions of
                label_img, but avoiding duplication of distances when the same distance-ij is the minimum distance both between region-i
                and the rest of the regions of label_img and of region-j and the rest of the regions of label_img.
                - Position-1 is a dictionary. Per each distance of the output at position-0, the coordinates of the pixel pair for which the
                distance has been calculated correspond to the key of the dictionary and they are linked to their distance as the dictionary value.
                - Position-2 is a dictionary. Let distance-ji be the minimum distance between region-j and the rest of the regions of label_img
                by connecting region-j to region-j. If it is not entered in the outputs of position-0 and position-1 because it correponds to
                the distance-ij, which is the minimum distance between region-i and the rest of the regions of label_img, and it connects region-i
                to region-j. The coordinates of the pixel pair connecting region-j to region-i for which distance-ji is entered in the
                ouput dictionary as the key of the dictionary and they are linked to their distance as the dictionary value.
        - if desired__distance=='max'.
            - if return_excluded_distances==False. the output is a tuple.
                - Position-0 is a list collecting, per each i region, the maximum euclidean distance with the rest of the regions of
                label_img, but avoiding duplication of distances when the same distance-ij is the maximum distance both between region-i
                and the rest of the regions of label_img and of region-j and the rest of the regions of label_img.
                - Position-1 is a dictionary. Per each distance of the output at position-0, the coordinates of the pixel pair for which the
                distance has been calculated correspond to the key of the dictionary and they are linked to their distance as the dictionary value.
            - if return_excluded_distances==True. the output is a tuple.
                - Position-0 is a list collecting, per each i region, the maximum euclidean distance with the rest of the regions of
                label_img, but avoiding duplication of distances when the same distance-ij is the maximum distance both between region-i
                and the rest of the regions of label_img and of region-j and the rest of the regions of label_img.
                - Position-1 is a dictionary. Per each distance of the output at position-0, the coordinates of the pixel pair for which the
                distance has been calculated correspond to the key of the dictionary and they are linked to their distance as the dictionary value.
                - Position-2 is a dictionary. Let distance-ji be the maximum distance between region-j and the rest of the regions of label_img
                by connecting region-j to region-j. If it is not entered in the outputs of position-0 and position-1 because it correponds to
                the distance-ij, which is the maximum distance between region-i and the rest of the regions of label_img, and it connects region-i
                to region-j. The coordinates of the pixel pair connecting region-j to region-i for which distance-ji is entered in the
                ouput dictionary as the key of the dictionary and they are linked to their distance as the dictionary value.
        
        - if desired__distance=='meam', the output is a tuple.
            Position-0 is a list of length n and collecting, per each region-i of label_img, its average distance with the rest of the regions of lable_img.
            Position-1 is None.
    """
    # Make sure highpass_n_regions is equal or higher than 1
    assert highpass_n_regions>=1, "highpass_n_regions must be >=1 as at least 2 regions must be present to calculate a distance"

    # Make sure that if a label image is passed as an input its values are >=0.
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

    #get the coordinates of all the regions in img
    all_regions_coords = [re.coords for re in regprops_img]

    # Only continue the analysis if there are more than a highpass_n_regions number of regions in label_img
    if len(all_regions_coords)>highpass_n_regions:
        #Initialize an output list
        output_list = []

        #If desired__distance is min or max, initialize an output dictionary, to track coordinate pairs already included in the analysis.
        # This will also be output if desired__distance=='min' or desired__distance=='max'
        #Also initialize a dictionary to double check, in case, the excluded coordinates
        if desired__distance=='min' or desired__distance=='max':
            output_dictionary_coords_dist = {}
            excluded_coordinates_dist_dict = {}

        #Iterate through the coordinates of the regions of img:
        for pos, r in enumerate(all_regions_coords):

            #Remove r coordinates from the general list
            all_regions_coords_copy = all_regions_coords.copy() #Re-initialize the global list of coordinates at each iteration, and copy it so that it is not modified
            all_regions_coords_copy.pop(pos)

            #Pool all the coordinates other than r in a unique list
            pooled_all_reg_all_regions_coords = []
            for r2 in all_regions_coords_copy:
                pooled_all_reg_all_regions_coords = pooled_all_reg_all_regions_coords + list(r2)
            #Re-transform pooled coordinates in a numpy array
            pooled_all_reg_coords_arr = np.asarray(pooled_all_reg_all_regions_coords)

            #Get the min/max/mean distance of r to pixels of the rest of regions within the input image, and the coordinates of the pixel pair
            wanted_distance, pixels_coords = get_euclidean_distances(r, pooled_all_reg_coords_arr, desired_distance=desired__distance)

            #If the min or max distances are calculated, avoid duplicating pixels pairs before collecting the distance in the
            #output list and collection dictionary
            if desired__distance=='min' or desired__distance=='max':
                if ((tuple(pixels_coords[0]), tuple(pixels_coords[1])) in output_dictionary_coords_dist) or ((tuple(pixels_coords[1]), tuple(pixels_coords[0])) in output_dictionary_coords_dist):
                    #Collect exlcuded coordinate pairs so that they could be also returned for checking purposes
                    excluded_coordinates_dist_dict[(tuple(pixels_coords[0]), tuple(pixels_coords[1]))]=wanted_distance
                else:
                    output_dictionary_coords_dist[(tuple(pixels_coords[0]), tuple(pixels_coords[1]))]=wanted_distance
                    output_list.append(wanted_distance)
            #If mean distance is calculated, there is no possibility of duplicating the distance
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

    # return a tuple with val_n_regions_nopass both in position 0 and position 1 if there are not more than highpass_n_regions number of regions in label_image
    else:
        return val_n_regions_nopass, val_n_regions_nopass

