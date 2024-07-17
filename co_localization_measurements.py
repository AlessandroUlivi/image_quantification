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
    - label_img_1_thres. Int or float. Only applies when transform_to_label_img is True. Default 0. Defines the highpass threshold to distinguish pixels of interest from background
    in label_img_1. Pixels whose value is >label_img_1_thres are considered pixels of interest. The rest of the pixels are considered background.
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
        assert np.min(label_img_1)==0, 'label_img_1 must have background values set to 0 if a label image is provided'
        assert np.max(label_img_1)>0, 'label_img_1 must have label region values >0 if a label image is provided'

    #Copy the input images - threshold that binary_mask_target using binary_mask_target_thres set it to values 1 and 0, where 1s are assumed to be the pixels of interest
    label_img_1_copy = label_img_1.copy()
    binary_mask_target_copy = np.where(binary_mask_target>binary_mask_target_thres, 1,0)

    #Transform input images to label images if transform_to_label_img is set to True
    if transform_to_label_img:
        #Theshold the input image using label_img_1_thres and set it to values 1 and 0, where 1s are assumed to be the pixels of interest
        label_img_1_copy_01 = np.where(label_img_1_copy>label_img_1_thres, 1,0)
        img_1_i = label(label_img_1_copy_01)
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


def count_number_of_overlapping_regions(arr_1_tot, arr_2_part, intersection_threshold=0, ro_i__mask=None, transform__to_label_img_arr_1=False,
                                        transform__to_label_img_arr_2=False, arr_1_tot_thres=0, arr_2_part_thres=0, return_regions=False,
                                        return_intersection_arrays=False, output_arr_loval=0, output_arr_highval=255, output_arr_dtype=np.uint8):
    """
    Returns the counts of regions of arr_1_tot which overlap with regions of arr_2_part. If return_regions is set to True, the regions properties are returned in a dictionary linking
    arr_1_tot regions with their overlapping regions in arr_2_part. If return_intersection_arrays is True, an array is returned per arr_1_tot and arr_2_parts containing all the regions
    sharing some overlap, as binary masks.

    Inputs:
    - arr_1_tot. ndarray. Binary mask or label image. The regions of this array are analysed for their overlap with regions of arr_2_part. The function returns in position 0,
    the number of the regions of this array which have 0, 1, 2, 3, 4 or more overlapping regions in arr_2_part.
    If label image, background pixels must be set to value 0 and pixels of interst to positve values.
    Note that the function relies on the use of skimage.measure.regionprops (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
    which relies on the use of label images. It is possible to transform a binary mask to a label image by setting transform__to_label_img_arr_1=True. If
    transform__to_label_img_arr_1=True, the array will be applied a highpass filtering process where pixels whose values is >arr_1_tot_thres (default 0) are set to 1
    and the remaining pixels are set to 0. The output array of this first step will then be transformed into a label image
    (refer to https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label).
    NOTE: Because a binary mask can be considered a label image if a single region is present, it is not mandatory to set transform__to_label_img_arr_1=True when a binary mask is passed.
    - arr_2_part. ndarray of the same shape of arr_1_tot. Binary mask or label image.
    The function returns in position 0 how many regions of arr_1_tot have 0, 1, 2, 3, 4 or more overlapping regions in this array.
    If label image, background pixels must be set to value 0 and pixels of interst to positve values.
    Note that the function relies on the use of skimage.measure.regionprops (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
    which relies on the use of label images. It is possible to transform a binary mask to a label image by setting transform__to_label_img_arr_2=True. If
    transform__to_label_img_arr_2=True, the array will be applied a highpass filtering process where pixels whose values is >arr_2_part_thres (default 0) are set to 1
    and the remaining pixels are set to 0. The output array of this first step will then be transformed into a label image
    (refer to https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label).
    - ro_i__mask. ndarray of the same shape of arr_1_tot. Optional. If provided, restricts the analysis to a region of interest. The region of interest is defined by the
    pixels in ro_i__mask whose intensity value is >0. NOTE: although not strictly required, the function was conceptualized for the use of binary masks for ro_i__mask.
    - transform__to_label_img_arr_1. Bool. Optional. Default False. If True, transforms arr_1_tot to a label image where pixels of interest are pixels whose intensity value
    is >arr_1_tot_thres.
    - transform__to_label_img_arr_2. Bool. Optional. Default False. If True, transforms arr_2_part to a label image where pixels of interest are pixels whose intensity value
    is >arr_2_part_thres.
    - arr_1_tot_thres. int or float. Optional, only applies if transform__to_label_img_arr_1=True. Default 0. Sets the highpass filter threshold to destinguish pixels of interest
    from background pixels when transforming arr_1_tot into a label image.
    - arr_2_part_thres. int or float. Optional. Default 0, only applies if transform__to_label_img_arr_2=True. Sets the highpass filter threshold to destinguish pixels of interest
    from background pixels when transforming arr_2_part into a label image.
    - return_regions. Bool. Optional. Default False. If True, a dictionary joining region-properties of regions of arr_1_tot to the region-properties of their overlapping regions in
    arr_2_part. Refer to outputs (position 1) for a detailed description.
    - return_intersection_arrays. Bool. Optional.  Default False. If True, returns 2 arrays of the same shape of arr_1_tot. Both arrays are binary masks. They contain, respectively, the
    regions of arr_1_tot which have at least one overlapping region in arr_2_part, and the regions of arr_2_part which have at least one overlapping region in arr_1_tot.
    The low and high values of the arrays can be specificed using, respectively, output_arr_loval and output_arr_highval. The data type can be specified using output_arr_dtype.
    - output_arr_loval. float or int. Optional, only applies if return_intersection_arrays=True. Default 0. Speficies the low value of the binary masks returned if
    return_intersection_arrays=True.
    - output_arr_highval. float or int. Optional, only applies if return_intersection_arrays=True.  Default 255. Speficies the high value of the binary masks returned if
    return_intersection_arrays=True.
     - output_arr_dtype. data-type. Optional, only applies if return_intersection_arrays=True. Default np.uint8. Speficies the data type of the binary masks returned if
     return_intersection_arrays=True.


    Outputs. tuple.
    I will call the individual regions in arr_1_tot as reg_1[i] and individual regions in arr_2_part as reg_2[j].
    - if return_regions=False and return_intersection_arrays=False.
        - position 0. dict. Keys are the number of overlapping reg_2[j] found per each individual reg_1[i]. Values are the number of times it is observed such number of reg_2[j]
        overlapping to reg_1[i]. So, for example, when key=0 and value=3 it means that 3 reg_1[i] have 0 overlapping reg_2[j].
        - position 1. None.
        - position 2. None.
        - position 3. None.
    
     - if return_regions=True and return_intersection_arrays=False.
        - position 0. dict. Keys are the number of overlapping reg_2[j] found per each individual reg_1[i]. Values are the number of times it is observed such number of reg_2[j]
        overlapping to reg_1[i]. So, for example, when key=0 and value=3 it means that 3 reg_1[i] have 0 overlapping reg_2[j].
        - position 1. dict. Keys are individual reg_1[i] of arr_1_tot as strings with progressive numbering. There is 1 key per reg_1[i] and the total number of keys correspond to the
        total number of reg_1[i] in arr_1_tot. Values are sub-dictionaries. Keys of sub-dictionaries are either 'arr_1' (string) or 'arr_2' (string) to define if the associated values
        refer to, respectively, arr_1_tot or arr_2_part. Values of sub-dictionaries are lists. The list linked to 'arr_1' contains 1 single element, which corresponds to
        the region-properties (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops) of reg_1[i]. The list linked to 'arr_2' contains one
        element per each reg_2[j] which is found to overlap with reg_1[i]. Each element corresponds to the region-properties
        (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops) of reg_2[j].
        - position 2. None.
        - position 3. None.
    
     - if return_regions=False and return_intersection_arrays=True.
        - position 0. dict. Keys are the number of overlapping reg_2[j] found per each individual reg_1[i]. Values are the number of times it is observed such number of reg_2[j]
        overlapping to reg_1[i]. So, for example, when key=0 and value=3 it means that 3 reg_1[i] have 0 overlapping reg_2[j].
        - position 1. None.
        - position 2. ndarry of same shape of arr_1_tot. Binary array. Per each reg_1[i], the pixels of the region are set to output_arr_highval if the region has at least one
        overlappy reg_2[j], to output_arr_lowval otherwise. The data type of the array is set by output_arr_dtype.
        - position 3. ndarry of same shape of arr_1_tot. Binary array. Per each reg_2[j], the pixels of the region are set to output_arr_highval if the region has at least one
        overlappy reg_1[i], to output_arr_lowval otherwise. The data type of the array is set by output_arr_dtype.
        
    - if return_regions=True and return_intersection_arrays=True.
        - position 0. dict. Keys are the number of overlapping reg_2[j] found per each individual reg_1[i]. Values are the number of times it is observed such number of reg_2[j]
        overlapping to reg_1[i]. So, for example, when key=0 and value=3 it means that 3 reg_1[i] have 0 overlapping reg_2[j].
        - position 1. dict. Keys are individual reg_1[i] of arr_1_tot as strings with progressive numbering. There is 1 key per reg_1[i] and the total number of keys correspond to the
        total number of reg_1[i] in arr_1_tot. Values are sub-dictionaries. Keys of sub-dictionaries are either 'arr_1' (string) or 'arr_2' (string) to define if the associated values
        refer to, respectively, arr_1_tot or arr_2_part. Values of sub-dictionaries are lists. The list linked to 'arr_1' contains 1 single element, which corresponds to
        the region-properties (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops) of reg_1[i]. The list linked to 'arr_2' contains one
        element per each reg_2[j] which is found to overlap with reg_1[i]. Each element corresponds to the region-properties
        (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops) of reg_2[j].
        - position 2. ndarry of same shape of arr_1_tot. Binary array. Per each reg_1[i], the pixels of the region are set to output_arr_highval if the region has at least one
        overlappy reg_2[j], to output_arr_lowval otherwise. The data type of the array is set by output_arr_dtype.
        - position 3. ndarry of same shape of arr_1_tot. Binary array. Per each reg_2[j], the pixels of the region are set to output_arr_highval if the region has at least one
        overlappy reg_1[i], to output_arr_lowval otherwise. The data type of the array is set by output_arr_dtype.
    """
    
    assert isinstance(return_regions, bool), "return_region is boolean. Must't be either True or False"
    assert isinstance(return_intersection_arrays, bool), "return_intersection_arrays is boolean. Must't be either True or False"

    #Make sure that if a label image is provided, the background pixels are set to 0 and the label pixels have positive values
    if not transform__to_label_img_arr_1:
        assert np.min(arr_1_tot)==0, 'arr_1_tot must have background values set to 0 if a label image is provided'
        assert np.max(arr_1_tot)>0, 'arr_1_tot must have label region values >0 if a label image is provided'
    #Add a warning if transform__to_label_img_arr_1 is True but a non-binary input is provided
    else:
        if len(np.unique(arr_1_tot))>2:
            print("warning, trasforming arr_1_tot in a label image, but arr_1_tot is a non-binary mask")


    #Make sure that if a label image is provided, the background pixels are set to 0 and the label pixels have positive values
    if not transform__to_label_img_arr_2:
        assert np.min(arr_2_part)==0, 'arr_2_part must have background values set to 0 if a label image is provided'
        assert np.max(arr_2_part)>0, 'arr_2_part must have label region values >0 if a label image is provided'
    #Add a warning if transform__to_label_img_arr_1 is True but a non-binary input is provided
    else:
        if len(np.unique(arr_2_part))>2:
            print("warning, trasforming arr_2_part in a label image, but arr_2_part is a non-binary mask")

    #Transform input images to label images if transform__to_label_img is set to True
    if transform__to_label_img_arr_1:
        #Copy the input images - threshold them using arr_1_tot_thres and arr_2_part_thres, set their values 1 and 0, where 1s are assumed to be the pixels of interest
        arr_1_tot_01 = np.where(arr_1_tot>arr_1_tot_thres, 1,0)
        label_arr_1_tot = label(arr_1_tot_01)
    else:
        label_arr_1_tot = arr_1_tot.copy()
    
    #Transform input images to label images if transform__to_label_img is set to True
    if transform__to_label_img_arr_2:
        #Copy the input images - threshold them using arr_1_tot_thres and arr_2_part_thres, set their values 1 and 0, where 1s are assumed to be the pixels of interest
        arr_2_part_01 = np.where(arr_2_part>arr_2_part_thres, 1,0)
        label_arr_2_part = label(arr_2_part_01)
    else:
        label_arr_2_part = arr_2_part.copy()

    #Set values outside roi to 0 (background) if ro_i__mask is provided
    if hasattr(ro_i__mask, "__len__"):
        ro_i__mask_copy = ro_i__mask.copy()
        arr_1_tot_to_proc = np.where(ro_i__mask_copy>0, label_arr_1_tot, 0)
        arr_2_part_to_proc = np.where(ro_i__mask_copy>0, label_arr_2_part, 0)
    else:
        arr_1_tot_to_proc = label_arr_1_tot
        arr_2_part_to_proc = label_arr_2_part
    
    #Get regionprops for the regions of arr_1 and arr_2
    arr_1_tot_regionprops = regionprops(arr_1_tot_to_proc)
    arr_2_part_regionprops = regionprops(arr_2_part_to_proc)

    #If return_intersection_arrays is True, initialize zero arrays to be modified as outputs. These can be used to verify the porcess
    if return_intersection_arrays:
        output_arr_1_tot = np.zeros(arr_1_tot_to_proc.shape)
        output_arr_2_part = np.zeros(arr_2_part_to_proc.shape)
    
    #If return_regions is True, initialize a dictionary to link per each region of arr_1_tot, its properties and the properties of the intersecting regions in arr_2_part
    if return_regions:
        output_regions_intersection_dict = {}

    #Initialize the output dictionary where keys are number of intersected regions in arr_2_part, and values are the number of times such number of intersection has been observed
    output_dict = {}

    #Iterate through the regions of arr_1_tot
    for counter, r1 in enumerate(list(arr_1_tot_regionprops)):

        #If return_regions is True, update output_regions_intersection_dict by linking the counter of the region, to the coordinates of the region and to a
        # sub-dictionary eventually containg the coordinates of intersecting regions in arr_2_part
        if return_regions:
            output_regions_intersection_dict['arr_1_reg_'+str(counter)]={'arr_1':[r1], 'arr_2':[]}

        #Get the coordinates of the region - organize them in a list of tuples
        r1_coords = r1.coords
        r1_coords_as_listoftupl = [(cr1[0], cr1[1]) for cr1 in list(r1_coords)]

        #Initialize the counter of the number of regions in arr_2_part intersecting r1 in arr_1_tot
        r2_intersected = 0

        #Iterate through the regions of arr_2_part
        for r2 in arr_2_part_regionprops:

            #Get the coordinates of the arr_2_part's region - transform them in a list of tuples
            r2_coords = r2.coords
            r2_coords_as_listoftupl = [(cr2[0], cr2[1]) for cr2 in list(r2_coords)]

            #Get intersection of region coordinates and mask_2 positive-pixels coordinates
            r1_on_interescion_r2 = list(set(r1_coords_as_listoftupl).intersection(set(r2_coords_as_listoftupl)))

            #If there is an intersection (number of of coordinates matching higher than >intersection_threshold)
            if len(r1_on_interescion_r2)>intersection_threshold:

                #Update the intersection counting
                r2_intersected = r2_intersected+1

                #If return_intersection_arrays is True, unzip the coordinates of r1 and r2 and modify the output arrays by setting the r1 and r2 pixels to 255
                if return_intersection_arrays:
                    # Unzip the coordinates in individual lists for r1
                    unzipped_r1_coords = [list(tt33) for tt33 in zip(*r1_coords)]
                    output_arr_1_tot[unzipped_r1_coords[0], unzipped_r1_coords[1]] = 255

                    # #Unzip the coordinates in individual lists for r2
                    unzipped_r2_coords = [list(tt333) for tt333 in zip(*r2_coords)]
                    output_arr_2_part[unzipped_r2_coords[0], unzipped_r2_coords[1]] = 255
                
                #If return_regions is True, update output_regions_intersection_dict by adding the properties of the r2
                if return_regions:
                    output_regions_intersection_dict['arr_1_reg_'+str(counter)]["arr_2"].append(r2)

        #Update the output_dict
        if r2_intersected in output_dict:
            output_dict[r2_intersected] += 1
        else:
            output_dict[r2_intersected] = 1

    #If return_intersection_arrays is True, set the value range and dtype according to the desired output values
    if return_intersection_arrays:
        final_output_arr_1_tot = np.where(output_arr_1_tot>0, output_arr_highval, output_arr_loval).astype(output_arr_dtype)
        final_output_arr_2_part = np.where(output_arr_2_part>0, output_arr_highval, output_arr_loval).astype(output_arr_dtype)
    
    #Return what wanted
    if return_intersection_arrays==True and return_regions==True:
        return output_dict, output_regions_intersection_dict, final_output_arr_1_tot, final_output_arr_2_part
    elif return_intersection_arrays==True and return_regions==False:
        return output_dict, None, final_output_arr_1_tot, final_output_arr_2_part
    elif return_intersection_arrays==False and return_regions==True:
        return output_dict, output_regions_intersection_dict, None, None
    else:
        return output_dict, None, None, None