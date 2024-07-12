import numpy as np
from random import sample
from skimage.measure import label, regionprops
from utils import get_euclidean_distances

def measure_pixels_overlap(arr_1, arr_2_against, roi_mask=None, shuffle_times=0, n_px_thr_1=1, n_px_thr_2=0, val_threshold_arr_1=0, val_threshold_arr_2=0):
    """
    Given two arrays (arr_1, arr_2_against), the function measures the fraction of pixels>val_threshold_arr_1 of arr_1 which are concomitantily >val_threshold_arr_2 in arr_2_against.
    val_threshold_1 and val_threshold_2 are, by default, set to 0.

    In the comments within the function I will call the pixels>val_threshold_arr_1 of arr_1 as "target pixels 1" and the pixels>val_threshold_arr_2 of arr_2_against as "target pixels 2".

    Inputs:
    arr_1 and arr_2_against are arrays of the same shape.

    roi_mask is an optional parameter. If provided it must be a numpy array of the same shape as arr_1 and arr_2_against. It restricts the analysis only to a region of interst
    assumed to correspond to the positive pixels of roi_mask array.

    shuffle_times. int. >=0. Default 0. If >0, for as many times as indicated, the pixels of arr_1 and arr_2_against are randomly shuffled and the
    fraction of pixels>val_threshold_arr_1 of arr_1 which are concomitantily >val_threshold_arr_2 in arr_2_against is re-calculared. If roi_mask is provided, the shuffling is done
    only in the roi.

    n_px_thr_1 (int or float, must be >0) is an optional input specifing a minum number of pixels which have to be >val_threshold_1 in in arr_1 in order
    to calculate the measurement. The default value is set to 1 as when 0 pixels are >val_threshold_2 in arr_2_against the output would imply a division by 0.

    n_px_thr_2 (int or float) is an optional input specifing a minum number of pixels which have to be >val_threshold_2 in in arr_2_against in order
    to calculate the measurement. The default value is set to 0.

    val_threshold_arr_1 and val_threshold_arr_2 are an int or float.

    Outputs:
    If either the number of pixels>val_threshold_1 in in arr_1 is lower than n_px_thr_1 or the number of pixels>val_threshold_2 in in arr_2_against is lower than n_px_thr_2, the output
    is None.
    Else:
        If shuffle times is set to 0, the output is a tuple, with, in position 0, the fraction of the pixels>val_threshold_1 of arr_1 which are concomitantly >val_threshold_2
        in arr_2_against for the input arr_1 and arr_2_against arrays. In position 1 is None.
        If shuffle times is > 0, the output is a tuple, with, in position 0, the fraction of the pixels>val_threshold_1 of arr_1 which are concomitantly >val_threshold_2
        in arr_2_against for the input arr_1 and arr_2_against arrays. In position 1 is a list. Inside the list the same measurement is done per each of the shuffle_times.

    NOTE: it is important to state that the function works on binary mask as well as non binary arrays with any pixel intensity value. However it was conceptualized for the use
    on binary masks where pixels of interests are indentified for their positive values.
    """
    
    #Check that n_px_thr_1 is >0, as values <=0 the could lead to 0 target pixels 1 and division by 0
    assert n_px_thr_1>0, "n_px_thr_1 must be higher than 0"

    #Copy the inputs arrays
    arr_1_copy = arr_1.copy()
    arr_2_against_copy = arr_2_against.copy()

    #Binarize arr_1 and arr_2_against using, respectively, val_threshold_arr_1 and val_threshold_arr_2 has highpass thresholds. Set the resulting arrays with values 0 and 1.
    arr_1_binary = np.where(arr_1_copy>val_threshold_arr_1, 1,0)
    arr_2_against_binary = np.where(arr_2_against_copy>val_threshold_arr_2, 1,0)

    #If an roi is provided, restrict the analysis to the roi
    if hasattr(roi_mask, "__len__"):
        roi_mask_copy = roi_mask.copy()
        arr_1_to_quantify = arr_1_binary[roi_mask>0]
        arr_2_against_to_quant = arr_2_against_binary[roi_mask>0]
    else:
        arr_1_to_quantify = arr_1_binary
        arr_2_against_to_quant = arr_2_against_binary
    
    # print(arr_1_to_quantify.shape)
    # print(arr_2_against_to_quant.shape)

    #Get the total number of target pixels 1 in arr_1
    arr_1_target_n = np.sum(arr_1_to_quantify)
    # print(arr_1_target_n)

    #Get the total number of target pixels 2 in arr_2_against
    arr_2_against_target_n = np.sum(arr_2_against_to_quant)
    # print(arr_2_against_target_n)

    # Continue with the measurement if the numbers of target pixels 1 and targer pixels 2 are respectively higher than n_px_thr_1 and n_px_thr_2
    if arr_1_target_n<n_px_thr_1 or arr_2_against_target_n<n_px_thr_2:
        print("there are less than ", n_px_thr_1, " target pixels in arr_1, or less than", n_px_thr_2, " target pixels in arr_2_against")
        print("target pixels in arr_1 are: ", arr_1_target_n)
        print("target pixels in arr_2_against are: ", arr_2_against_target_n)
        return
    else:
        #Binarize arr_2_against by setting target pixels 2 to 1 if they are corresponded by target pixels 1 in arr_1, to 0 otherwise
        arr_1_arr_2_intersection = np.where(arr_1_to_quantify>0, arr_2_against_to_quant, 0)

        # print("===")
        #Count the number of target pixels 2 in arr_2_against which are corresponded by target pixels 1 in arr_1
        target_2_target_1_double_positive = np.sum(arr_1_arr_2_intersection)
        # print(target_2_target_1_double_positive)
        #Calculate the fraction of target pixels 1 which are also target pixels 1
        fract_double_target_on_target_1= target_2_target_1_double_positive/arr_1_target_n
        # print(fract_double_target_on_target_1)

        if shuffle_times==0:
            return fract_double_target_on_target_1, None
        else:

            #Initialize a list to collect shuffling results
            shuffling_results = []

            #Iterate through the number of shuffling times
            for i in range(shuffle_times):

                #randomly shuffle arr_1 and arr_2_against
                list_1_shuff = sample(list(arr_1_to_quantify.flatten()), k=len(list(arr_1_to_quantify.flatten())))
                list_2_against_shuff = sample(list(arr_2_against_to_quant.flatten()), k=len(list(arr_2_against_to_quant.flatten())))

                #Transform shuffled lists into numpy arrays
                arr_1_shuff = np.asarray(list_1_shuff)
                arr_2_against_shuff = np.asarray(list_2_against_shuff)

                #Binarize arr_2_against_shuff by setting target pixels 2 to 1 if they are corresponded by target pixels 1 in arr_1_shuff, to 0 otherwise
                shuff_arr_1_arr_2_intersection = np.where(arr_1_shuff>0, arr_2_against_shuff, 0)

                # print("===")
                #Count the number of target pixels 2 in arr_2_against_shuff which are corresponded by target pixels 1 in arr_1_shuff
                target_2_target_1_double_positive_shuff = np.sum(shuff_arr_1_arr_2_intersection)
                # print(target_2_target_1_double_positive_shuff)
                #Calculate the fraction of target pixels 1 which are also target pixels 1
                fract_double_target_on_target_1_shuff = target_2_target_1_double_positive_shuff/arr_1_target_n
                # print(fract_double_target_on_target_1_shuff)

                #Add the result to the collection list
                shuffling_results.append(fract_double_target_on_target_1_shuff)
            
            return fract_double_target_on_target_1, shuffling_results


def measure_regions_euclidean_distances(label_img_1, binary_mask_target, roi__mask=None, desired__distance='min', transform_to_label_img=False, label_img_1_thres=0,
                                        binary_mask_target_thres=0):
    """
    Returns the eucledean distance between the regions of an input image (label_img_1) and the regions of a target image (binary_mask_target).

    Inputs:
    - label_img_1. ndarray. It can either be a label image (an image where pixels of separate regions are assigned the same value and a unique value is assigned to each separate region)
    or a binary image. If a label image is provided, the values must be >=0 and pixels of value 0 are assumed to correspond to the background. If a binary image is given,
    the parameter transform_to_label_img must be set to True in order for the function to transfom it in a label image using skimage.measure.label method
    (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label). If a binary mask is provided it will be firstly rescaled in the 0-1 range using the
    label_img_1_thres parameter. Pixels with intensity values >label_img_1_thres will be set to 1 and considered pixels of interest, while the rest will be
    set to 0 and considered background. The default value for label_img_1_thres is 0. NOTE: a binary mask can be considered a label image if only a single
    region is present.
    - binary_mask_target. ndarray of the same shape of label_img_1. Pixels of interest are assumed to be pixels whose value is >binary_mask_target_thres (default 0).
    - roi_mask. Optional parameter. ndarray of the same shape of label_img_1. Binary mask. When provided, the analysis will be restricted to the region of interest indicated by roi_mask.
    It is assumed that the region of interest corresponds to pixels in roi_mask with value>0.
    - desired_distance. String. Three choices are possible: 'min' (default), 'max', 'mean'. The parameter is passed to the function get_euclidean_distances (within utils.py). If 'min'
    the minimum distances between regions of label_img_1 and regions of binary_mask_target are measured. If 'max' the maximum distances between regions of label_img_1
    and regions of binary_mask_target are measured. If 'mean' the mean distance between between regions of label_img_1 and regions of binary_mask_target is measured.
    - transform_to_label_img. Bool (defaul False). Specifies if label_img_1 should be transformed to a label image. It must be True if label_img_1 is not a label image.
    - binary_mask_target_thre. Int or float. Pixels whose value is >binary_mask_target_thre are considered pixels of interest in binary_mask_target.

    Outputs:
    I will use i to refer to the i-th region of label_img_1. n is the total number of separate regions in label_img_1.
    - if desired__distance=='min', the output is a tuple. Position-0 is a list of length n collecting, per each i region, the minimum euclidean distance with regions of
    binary_mask_target. Position-1 is a dictionary with n entries, each entry corresponding to a i region. Each entry links the coordinates of the pixel pair
    (key is the pixel in the i region and value is the pixel among the pixels of interest of binary_mask_target) for which the minimum distance between region i and regions of
    binary_mask_target have been calculated.
    - if desired__distance=='max', the output is a tuple. Position-0 is a list of length n collecting, per each i region, the maximum euclidean distance with regions of
    binary_mask_target. Position-1 is a dictionary with n entries, each entry corresponding to a i region. Each entry links the coordinates of the pixel pair
    (key is the pixel in the i region and value is the pixel among the pixels of interest of binary_mask_target) for which the maximum distance between region i and regions of
    binary_mask_target have been calculated.
    - if desired__distance=='meam', the output is a tuple. Position-0 is a float value indicating the average distance between regions of label_img_1 and regions of binary_mask_target.
    Position-1 is None.
    """
    
    assert len(np.unique(binary_mask_target))==2, 'binary_mask_target must be a binary mask'
    if not transform_to_label_img:
        assert np.min(label_img_1)==0, 'label_img_1 must have background values set to 0'
        assert np.max(label_img_1)>0, 'label_img_1 must have label region values >0'

    #Copy the input images - make sure that binary_mask_target has values 1 and 0, where 1s are assumed to be the pixels of interest
    label_img_1_copy = label_img_1.copy()
    binary_mask_target_copy = np.where(binary_mask_target>binary_mask_target_thres, 1,0)

    #Transform input images to label images if transform_to_label_img is set to True
    if transform_to_label_img:
        rescaled_label_img_1_copy = np.where(label_img_1_copy>label_img_1_thres, 1,0)
        img_1_i = label(rescaled_label_img_1_copy)
    else:
        img_1_i = label_img_1_copy

    #Set values outside roi to 0 (background) if roi__mask is provided
    if hasattr(roi__mask, "__len__"):
        roi__mask_copy = roi__mask.copy()
        img_1 = np.where(roi__mask>0, img_1_i, 0)
        target_mask = np.where(roi__mask>0, binary_mask_target_copy, 0)
    else:
        img_1 = img_1_i
        target_mask = binary_mask_target_copy
    
    #get regionproperties of label_img_1 and binary_mask_target
    regprops_img_1 = regionprops(img_1)
    regionprops_target_mask = regionprops(target_mask)
    # print("img_1", type(regprops_img_1), len(regprops_img_1))
    # print("target", type(regionprops_target_mask), len(regionprops_target_mask))

    #Get the coordinates of the pixels in binary_mask_target
    target_mask_coords_i = regionprops_target_mask[0].coords
    # print(target_mask_coords_i)
    # # transform coordinates of the pixels in the binary_mask_target in a list of tuples
    # target_mask_coords = [tuple(tc) for tc in target_mask_coords_i]
    # print(target_mask_coords)

    #Initialize an output list
    output_list = []

    #If desired__distance is min or max, initialize an output dictionary, linking per each region of img_1, the coordinates of the pixels for which the distance was calculated.
    if desired__distance=='min' or desired__distance=='max':
        output_coords_dict = {}

    #Iterate through the regions of img_1:
    for r1 in regprops_img_1:

        #Get the coordinates of the pixels of r1 and transform them in a list of tuples
        r1_coords_i = r1.coords
        # r1_coords = [tuple(r1c) for r1c in r1_coords_i]

        #Get the min/max/mean distance of r1 to pixels of target_mask, and the coordinates of the pixel pair
        wanted_distance, pixels_coords = get_euclidean_distances(r1_coords_i, target_mask_coords_i, desired_distance=desired__distance)

        #Add measured distance to output list
        output_list.append(wanted_distance)

        #If desired__distance is min or max,link the coordinates of the pixels for which the distance was calculated in output_coords_dict
        if desired__distance=='min' or desired__distance=='max':
            output_coords_dict[tuple(pixels_coords[0])]=tuple(pixels_coords[1])

    #If desired__distance is min or max, return distances list and dictionary of paired pixels coordinates. Return the distances list and None if  desired__distance is mean
    if desired__distance=='min' or desired__distance=='max':
        return output_list, output_coords_dict
    else:
        return output_list, None


def count_number_of_overlapping_regions():
    return

