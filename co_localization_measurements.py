import numpy as np
from random import sample

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
        in arr_2_against for the input arr_1 and arr_2_against arrays.
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
            return fract_double_target_on_target_1,
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




