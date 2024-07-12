import numpy as np
from skimage.measure import label

def count_regions_number(input_arr, roi_mask=None, threshold_input_arr=0, threshold_roi_mask=0):
    """
    returns the number of separated regions in an input array.

    Inputs:
    - input_arr. ndarray. The array undergoes a binarization step. Pixels whose value is >threshold_input_arr are considered pixels of interest, the rest background.
    - roi_mask. Optional. ndarray of the same shape of input_arr. If provided, restricts the analysis to a region of interest identified by pixels whose value is >threshold_roi_mask.
    - threshold_input_arr. int or float. Default 0. Defines the highpass threshold to distinguish pixels of interest from background pixels in input_arr.
    - threshold_roi_mask. int or float. Default 0. Only applies if roi_mask is provided. Defines the highpass threshold to distinguish pixels of interest from background pixels
    in roi_mask.

    Output: float. The number of separated regions in input array. The identification relies on the transformation of the input array in a label image and the following identification
    of the number of labelled regions. Refer to https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label for the labelling process.
    """
    #Copy the input array. Binarize it using threshold_input_arr and set the values to 0 and 1.
    input_arr_01 = np.where(input_arr>threshold_input_arr, 1, 0)

    #If roi_mask is provided, copy it, binarize it using threshold_roi_mask, set the pixels value to 0 and 1, and use it to set to 0 the pixels in input_arr corresponded by background
    #pixels in roi_mask
    if hasattr(roi_mask, "__len__"):
        roi_mask_01 = np.where(roi_mask>threshold_roi_mask, 1,0)
        input_arr_to_count = np.where(roi_mask_01>0, input_arr_01, 0)
    else:
        input_arr_to_count = input_arr_01
    
    #Label input_arr and get the number of labelled regions - reg_number is already correctly returning only the number of labelled regions, thus excluding the background
    label_input_arr, reg_number = label(input_arr_01, return_num=True)

    return reg_number
