import numpy as np
import pandas as pd
from scipy.stats import sem
from co_localization_measurements import measure_pixels_overlap, measure_regions_euclidean_distances, count_number_of_overlapping_regions
from counting_measurements import count_regions_number
from geometric_measurements import get_mask_area, get_areas_of_regions_in_mask
from topological_measurements import get_convex_hull_fraction, measure_regions_euclidean_distances_within_array
from utils import match_arrays_dimensions


# measure_pixels_overlap,
# measure_regions_euclidean_distances,
# count_number_of_overlapping_regions
# get_convex_hull_fractions
# measure_regions_euclidean_distances_within_array


def quantify_channels(channels_array, channels_axis=0, roi_mask_array=None, analysis_axis=None, shuffle_times=0, no_quantification_valu_e=np.nan,
                      channels_binarization_thresholds=0, transform_to_label_img=False, get_mask_area_val_4zero_regionprops=0,
                      count_regions_number_threshold_roi_mask=0, n_of_region_4areas_measure=0, reg_eucl_dist_within_arr_val_n_regions_nopass=1,
                      min_px_over_thresh_common=-1, measure_pixels_overlap_n_px_thr_1=1, measure_pixels_overlap_n_px_thr_2=0,
                      count_n_overl_reg_intersection_threshold=None, conv_hull_fract_px_thre_arr_1=3, conv_hull_fract_px_thre_arr_2=3,
                      get_conv_hull_fract_arr1_NOpass_arr2_pass_v=0.0, get_conv_hull_fract_arr2_NOpass_v=np.nan):
    """
    seems to work with 3 limitations: 1) when providing the thresholds they array cannot contain axis of size 1. 2) when channel_axis/analysis_axis is in position 0 it can't
    be indicated using negative number indexing (for example -10). 3) If a custom array of thresholds is provided and multiple thresholds are required (e.g. for comparison
    functions, the multiple thresholds must be in position -1)
    - roi_mask_array can be different for the channels. At least 1 axis mutch match channels_arrays. The matching axis must be in the correct position.
    - min_px_over_thresh_common. the number of o pixels both channels must pass to continue with paired measurements.
    NOTE than when a tuple or a list is passed as a threshold this is interpreted as a multi-threshold, not as individual thresholds for the different channels.
    NOTE WELL THAT transform_to_label_img=False.
    NOTE get_conv_hull_fract_arr1_NOpass_arr2_pass_v and get_conv_hull_fract_arr2_NOpass_v can't be changed tailored on the specific channel and array of analysis axis.
    """
    #=========================================
    #=========  SUPPORTING FUNCTIONS =========
    def modify_dictionary(result_valu_e, dict2modify, root_key_name, channel_1_number=None, channel_2_number=None):

        if isinstance(result_valu_e, list) or isinstance(result_valu_e, tuple):
            for i,v in enumerate(result_valu_e):
                if channel_1_number!=None:
                    if channel_2_number!=None:
                        result_name = f"{root_key_name}_ch_{channel_1_number}_ch_{channel_2_number}_iter_{i}"
                    else:
                        result_name = f"{root_key_name}_ch_{channel_1_number}_iter_{i}"
                    
                    if result_name not in dict2modify:
                        dict2modify[result_name]=[v]
                    else:
                        dict2modify[result_name].append(v)
                    print(result_name, len(dict2modify[result_name]))
                else:
                    result_name = f"{root_key_name}_iter_{i}"

                    if result_name not in dict2modify:
                        dict2modify[result_name]=[v]
                    else:
                        dict2modify[result_name].append(v)
                    print(result_name, len(dict2modify[result_name]))
        else:
            if channel_1_number!=None:
                if channel_2_number!=None:
                    result_name = f"{root_key_name}_ch_{channel_1_number}_ch_{channel_2_number}"
                else:
                    result_name = f"{root_key_name}_ch_{channel_1_number}"
                
                if result_name not in dict2modify:
                    dict2modify[result_name]=[result_valu_e]
                else:
                    dict2modify[result_name].append(result_valu_e)
                print(result_name, len(dict2modify[result_name]))
            else:
                result_name=root_key_name

                if result_name not in dict2modify:
                    dict2modify[result_name]=[result_valu_e]
                else:
                    dict2modify[result_name].append(result_valu_e)
                print(result_name, len(dict2modify[result_name]))

    def get_mean_median_std_sem_min_max_results(results_measurements, no_quantification_value=np.nan):
        """
        many of the output measurements are lists and the length of the list depends/corresponds to the number of quantified elements (e.g. regions).
        - if the list has 3 or more quantifications. Return the mean, median, standard deviation, standard error of mean, max and min values of the measurement list.
        - if the list has 2 quantifications. Return the mean, median, max and min values of the measurement list and return no_quantification_value for the standard deviation and standard
        error of means.
        - if the list has 1 quantification. Return the value for the mean, median, max and min and return no_quantification_value for the standard deviation and standard
        error of means.
        - the list has 0 quantifications. Return no_quantification_value for mean, median, standard deviation, standard error of mean, max and min.

        Inputs:
        - results_measurements. list of tuple.
        - no_quantification_value. any. Default np.nan.

        Output:
        tuple. Pos-0, number of quantified elements. Pos-1, mean of results_measurements. Pos-2, median of results_measurements. Pos-3, standard deviation of results_measurements.
        Pos-4, standard error of the means of results_measurements. Pos-5, min of results_measurements. Pos-6, max of results_measurements. 
        """
        if not isinstance(results_measurements,list) and not isinstance(results_measurements, tuple):
            print("results_measurements must be either a list or a tuple")
            return
        
        if len(results_measurements)>2:
            mean_results_measurements = np.mean(results_measurements)
            median_results_measurements = np.median(results_measurements)
            stdv_results_measurements = np.std(results_measurements)
            sem_results_measurements = sem(results_measurements)
            min_results_measurements = np.min(results_measurements)
            max_results_measurements = np.max(results_measurements)
        elif len(results_measurements)==2:
            mean_results_measurements = np.mean(results_measurements)
            median_results_measurements = np.median(results_measurements)
            stdv_results_measurements = no_quantification_value
            sem_results_measurements = no_quantification_value
            min_results_measurements = np.min(results_measurements)
            max_results_measurements = np.max(results_measurements)
        elif len(results_measurements)==1:
            mean_results_measurements = results_measurements[0]
            median_results_measurements = results_measurements[0]
            stdv_results_measurements = no_quantification_value
            sem_results_measurements = no_quantification_value
            min_results_measurements = results_measurements[0]
            max_results_measurements = results_measurements[0]
        else:
            mean_results_measurements = no_quantification_value
            median_results_measurements = no_quantification_value
            stdv_results_measurements = no_quantification_value
            sem_results_measurements = no_quantification_value
            min_results_measurements = no_quantification_value
            max_results_measurements = no_quantification_value
        return len(results_measurements), mean_results_measurements, median_results_measurements, stdv_results_measurements, sem_results_measurements, min_results_measurements, max_results_measurements
    
    def set_thresholds_2use(input_thresholds, channels_stac_k):
        """
        Inputs:
        - input_thresholds. None, int/float, tuple/list or ndarray.
        - channels_stac_k. ndarray.

        Outputs:
        - if input_thresholds is int/float. The output is an ndarray of the same shape of channels_stac_k and all values set to input_threshols.
        - if input_thresholds is tuple/list. The output is an ndarray of shape=channels_stac_k.shape+1. The extra dimension is in position -1 and its size corresponds to the
        length of input_thresholds. Each ndarray of shape=channels_stac_k.shape which is stacked along the -1 dimension, contains one of the values of input_thresholds.
        - if input_thresholds is ndarray. The output is input_thresholds. NOTE: it multiple thresholds are reported and input_thresholds is ndarray, the multiple threshold values
        must be indicated on axis -1.
        """
        if isinstance(input_thresholds, int) or isinstance(input_thresholds, float):
            return np.where(np.zeros(channels_stac_k.shape)==0, input_thresholds,input_thresholds)
        elif isinstance(input_thresholds, tuple) or isinstance(input_thresholds, tuple):
            return np.stack([np.where(np.zeros(channels_stac_k.shape)==0, input_thresholds[k],input_thresholds[k]) for k in range(len(input_thresholds))], axis=-1)
        else:
            return input_thresholds

    def split_thresholds_arrays(thr_array, split_axis, multi_thresholds=False):
        """
        Splits thr_array along split_axis. If multi_thresholds array is True, split_axis is reduced of 1 number.
        This function works in the context of set_thresholds_2use and quantify_channels. For certain quantification functions it is required to indicate multiple thresholds. These
        thresholds must be indicated in the -1 axis of the threshold array which is either the output of set_threshold_2use or directly input to quantify_channels. For this reason,
        when multiple thresholds are reported, the threshold array has an additional axis than channel_array (see quantify_channels inputs), in position -1. When channels_array and
        threshold arrays have to be split, the axis along which to do the split is referring to channel array and does not take into consideration the extra dimension of
        thresholds array. This results in a wrong indexing when the index is indicated using negative numbers. The present function compensates for this fact.
        """
        if multi_thresholds:
            if split_axis<0:
                split_axis_2use = split_axis-1
            else:
                split_axis_2use = split_axis
        else:
            split_axis_2use = split_axis
        
        return [np.squeeze(z) for z in np.split(thr_array, indices_or_sections=thr_array.shape[split_axis_2use], axis=split_axis_2use)]
    
    def get_threshold_from_list(array_threshold, multi_value_array=False, multi_value_axis=-1, get_a_single_value=True):
        """
        Given:
        array_threshold. ndarray.

        The function returns:
        - if multi_value_array==False (default).
            - if get_a_single_value==True. The function returns the average value of array_threshold, as an int or float.
            - if get_a_single_value==False. The function returns array_threshold.
        - if multi_value_array==True. The function splits array_threshold in sub-arrays along multi_value_axis (defaul is -1).
            - if get_a_single_value==True. The function returns a tuple with the average value of the of each sub-array of array_threshold along the multi_value_axis.
            - if get_a_single_value==True. The function return the sub-arrays obtained by splitting array_threshold along the multi_value_axis. The output dtype is a list.
        
        This function works in the context of set_thresholds_2use and quantify_channels. For certain quantification functions it is required to indicate multiple thresholds. These
        thresholds must be indicated in the -1 axis of the threshold array which is either the output of set_threshold_2use or directly input to quantify_channels. For this reason,
        when multiple thresholds are reported, the threshold array has an additional axis than channel_array (see quantify_channels inputs), in position -1. When the thresholds have
        to be retrieved, this function allows to retrieve each individual multi-threshold from the extra dimension, if required.
        NOTE:
        - although it is possible to change the axis where multi-thresholds are reported by changing multi_value_axis, the whole processing was conceptualize for having these
        values in the axis -1 and it hasn't been tested for different situations.
        - Although it is possible to retrieve an entire array instead of a single threshold value, the process is conceptualized for retrieving a single value and it hasn't been
        tested for different situations.
        """
        if multi_value_array:
            multi_thresholds_split = [np.squeeze(y) for y in np.split(array_threshold, indices_or_sections=array_threshold.shape[multi_value_axis], axis=multi_value_axis)]
            if get_a_single_value:
                return tuple([np.mean(mta) for mta in multi_thresholds_split])
            else:
                return multi_thresholds_split
        else:
            if get_a_single_value:
                return np.mean(array_threshold)
            else:
                return array_threshold

    #=======================================
    #=========  PREPARE INPUT DATA =========
    #Make sure that channels_axis and analysis axis are not the same axis
    assert channels_axis != analysis_axis, "channels_axis can't be the same of analysis_axis"

    #If arrays are split along an axis to analyses, because channels_axis will work after the splitting, reduce it of 1 unit to compensate for the recuded dimension
    if analysis_axis != None:
        if channels_axis>analysis_axis:
            channels_axis_2use = channels_axis-1
        else:
            channels_axis_2use = channels_axis
    else:
        channels_axis_2use = channels_axis

    #Copy input arrays
    channels_array_copy = channels_array.copy()

    #If roi_mask_array is provided make sure it is has the same shape of channels_array, else, copy the None
    if hasattr(roi_mask_array, "__len__"):
        roi_mask_array_copy_i = roi_mask_array.copy()
        roi_mask_array_2use = match_arrays_dimensions(roi_mask_array_copy_i, channels_array_copy)
    else:
        roi_mask_array_2use = roi_mask_array

    #=========================================================
    #=========  GET THRESHOLDS IN THE CORRECT FORMAT =========
    #Set binarization thresholds to 0 for all channels, if channels_binarization_thresholds is not provided. Use provided values othewise.
    ch_bin_thresh_2use = set_thresholds_2use(channels_binarization_thresholds, channels_stac_k=channels_array_copy)

    #Set to False transform_to_label_img in measure_regions_euclidean_distances if transform_to_label_img is not provided. Use the provided value otherwise
    transform_to_label_img_2use = set_thresholds_2use(transform_to_label_img, channels_stac_k=channels_array_copy)
    
    #Set val_4zero_regionprops in get_mask_area as 0 by defaut, if None is provided as input. Use the provided value otherwise.
    val_4zero_regionprops_2use = set_thresholds_2use(get_mask_area_val_4zero_regionprops, channels_stac_k=channels_array_copy)

    #Set threshold_roi_mask in count_regions_number as 0 by defaut, if None is provided as input. Use the provided value otherwise.
    threshold_roi_mask_2use = set_thresholds_2use(count_regions_number_threshold_roi_mask, channels_stac_k=channels_array_copy)

    #Set to 0 the highpass threshold for calculating the mean, median, min and max area of regions within a channel, if None is provided as input. Use the provided value otherwise.
    n_of_region_4areas_measure_2use = set_thresholds_2use(n_of_region_4areas_measure, channels_stac_k=channels_array_copy)

    #Set to 1 the highpass threshold for calculating distances within regions of a channel, if reg_eucl_dist_within_arr_val_n_regions_nopass is not provided.
    #Use the provided value otherwise
    reg_eucl_dist_within_arr_val_n_regions_nopass_2use = set_thresholds_2use(reg_eucl_dist_within_arr_val_n_regions_nopass, channels_stac_k=channels_array_copy)

    #Set to 0 the min number of pixels for proceeding with measurements, if min_px_over_thresh_common is not provided. Use the provided thresholds otherwise
    min_px_over_thresh_common_2use = set_thresholds_2use(min_px_over_thresh_common, channels_stac_k=channels_array_copy)
    
    #Set to 1 the min number of pixels of arr_1 for calculating pixels overlap in measure_pixels_overlap, if measure_pixels_overlap_n_px_thr_1 is not provided.
    # Use the provided thresholds otherwise
    measure_pixels_overlap_n_px_thr_1_2use = set_thresholds_2use(measure_pixels_overlap_n_px_thr_1, channels_stac_k=channels_array_copy)
    
    #Set to 0 the min number of pixels of arr_2_against for calculating pixels overlap in measure_pixels_overlap, if measure_pixels_overlap_n_px_thr_2 is not provided.
    # Use the provided thresholds otherwise
    measure_pixels_overlap_n_px_thr_2_2use = set_thresholds_2use(measure_pixels_overlap_n_px_thr_2, channels_stac_k=channels_array_copy)

    #Set to tuple of 0s of length=channel_axis'size * (channel_axis'size-1) intersection_threshold in count_number_of_overlapping_regions if count_n_overl_reg_intersection_threshold
    # is not provided. Use the provided tuple otherwise
    if count_n_overl_reg_intersection_threshold==None:
        default_intersection_thresholds = tuple([0 for chnll in range(channels_array_copy.shape[channels_axis]*channels_array_copy.shape[channels_axis])])
        count_n_overl_reg_intersection_threshold_2use = set_thresholds_2use(default_intersection_thresholds, channels_stac_k=channels_array_copy)
    else:
        count_n_overl_reg_intersection_threshold_2use = set_thresholds_2use(count_n_overl_reg_intersection_threshold, channels_stac_k=channels_array_copy)
    
    #Set to 3 the min number of pixels of arr_1 for calculating the convex hull fraction, if conv_hull_fract_px_thre_arr_1 is not provided. Use the provided thresholds otherwise
    conv_hull_fract_px_thre_arr_1_2use = set_thresholds_2use(conv_hull_fract_px_thre_arr_1, channels_stac_k=channels_array_copy)

    #Set to 3 the min number of pixels of arr_2 for calculating the convex hull fraction, if conv_hull_fract_px_thre_arr_2 is not provided. Use the provided thresholds otherwise
    conv_hull_fract_px_thre_arr_2_2use = set_thresholds_2use(conv_hull_fract_px_thre_arr_2, channels_stac_k=channels_array_copy)

    #==========================================
    #=========  INITIALIZE THE OUTPUT =========
    #Initialize a dictionary to be used to be used to form the output datafram
    measurements_dict = {}

    #==================================================
    #=========  USE ANALYSIS AXIS IF PROVIDED =========
    #If analysis axis is provided:
    if analysis_axis != None:
        
        #==============================================================================
        #=========  PREPARE ROI AND THRESHOLDS FOR ITERATION ON ANALYSIS AXIS =========
        #Split the roi_mask_array on the analysis_axis, if roi_mask_array is provided
        if hasattr(roi_mask_array, "__len__"):
            roi_mask_array_2use_1 = [np.squeeze(w) for w in np.split(roi_mask_array_2use,
                                                                         indices_or_sections=channels_array_copy.shape[analysis_axis],
                                                                         axis=analysis_axis)]
            # print("roi_mask after analysis axis split: ", len(roi_mask_array_2use_1), roi_mask_array_2use_1[0].shape)
        else:
            roi_mask_array_2use_1 = roi_mask_array_2use #which should be meaning None

        #Split threshold arrays on the analysis_axis
        ch_bin_thresh_2use_1 = split_thresholds_arrays(ch_bin_thresh_2use, split_axis=analysis_axis, multi_thresholds=False)
        transform_to_label_img_2use_1 = split_thresholds_arrays(transform_to_label_img_2use, split_axis=analysis_axis, multi_thresholds=False)
        val_4zero_regionprops_2use_1 = split_thresholds_arrays(val_4zero_regionprops_2use, split_axis=analysis_axis, multi_thresholds=False)
        threshold_roi_mask_2use_1 = split_thresholds_arrays(threshold_roi_mask_2use, split_axis=analysis_axis, multi_thresholds=False)
        n_of_region_4areas_measure_2use_1 = split_thresholds_arrays(n_of_region_4areas_measure_2use, split_axis=analysis_axis, multi_thresholds=False)
        reg_eucl_dist_within_arr_val_n_regions_nopass_2use_1 = split_thresholds_arrays(reg_eucl_dist_within_arr_val_n_regions_nopass_2use, split_axis=analysis_axis, multi_thresholds=False)
        min_px_over_thresh_common_2use_1 = split_thresholds_arrays(min_px_over_thresh_common_2use, split_axis=analysis_axis, multi_thresholds=False)
        measure_pixels_overlap_n_px_thr_1_2use_1 = split_thresholds_arrays(measure_pixels_overlap_n_px_thr_1_2use, split_axis=analysis_axis, multi_thresholds=False)
        measure_pixels_overlap_n_px_thr_2_2use_1 = split_thresholds_arrays(measure_pixels_overlap_n_px_thr_2_2use, split_axis=analysis_axis, multi_thresholds=False)
        count_n_overl_reg_intersection_threshold_2use_1 = split_thresholds_arrays(count_n_overl_reg_intersection_threshold_2use, split_axis=analysis_axis, multi_thresholds=True)
        conv_hull_fract_px_thre_arr_1_2use_1 = split_thresholds_arrays(conv_hull_fract_px_thre_arr_1_2use, split_axis=analysis_axis, multi_thresholds=False)
        conv_hull_fract_px_thre_arr_2_2use_1 = split_thresholds_arrays(conv_hull_fract_px_thre_arr_2_2use, split_axis=analysis_axis, multi_thresholds=False)

        #====================================================================================================
        #=========  INITIALIZE DICTIONARY COLLECTING RESULTS OF count_number_of_overlapping_regions =========
        #The output of count_number_of_overlapping_regions is a dictionary linking the amount of regions in channel-j overlapping to region-i of channel-i to the
        #number of times such number of overlapping regions has been observed. This output non fixed, it is thus impossible to establish a prioi how main columns in
        #the output dataframe will contain this quantification. The only way to do it is to collect all the output first and then re-arrange them at the end.
        #Here a dictionary is initialized to collect all these outputs
        count_number_of_overlapping_regions_coll_dict = {}

        #=================================================
        #=========  ITERATE ON THE ANALYSIS AXIS =========
        # Iterate through the analysis axis
        for ixd, idx_array in enumerate([np.squeeze(a) for a in np.split(channels_array_copy,
                                                                         indices_or_sections=channels_array_copy.shape[analysis_axis],
                                                                         axis=analysis_axis)]):
            # print("==="*3, ixd)
            #============================================
            #========= UPDATE OUTPUT DICTIONARY =========
            #Update measurements_dict, which will be used to form the output dataframe
            # modify_dictionary(result_valu_e=ixd, dict2modify=measurements_dict, root_key_name='axis_'+str(analysis_axis), channel_1_number=None, channel_2_number=None)
            
            #Also update count_number_of_overlapping_regions_coll_dict, to be used for adding the measurements of count_number_of_overlapping_regions to measurements_dict,
            #at the end of the iterations
            count_number_of_overlapping_regions_coll_dict[ixd]={}

            #===================================================================================
            #=========  PREPARE DATA, ROI AND THRESHOLDS FOR ITERATION ON CHANNEL AXIS =========
            #Get the individual channels array as a list
            ch_arrays_list = [np.squeeze(b) for b in np.split(idx_array, indices_or_sections=idx_array.shape[channels_axis_2use], axis=channels_axis_2use)]

            #Also get the individual channels arrays as a list for the roi_mask, if it is provided
            if hasattr(roi_mask_array, "__len__"):
                roi_mask_array_2use_2 = [np.squeeze(v) for v in np.split(roi_mask_array_2use_1[ixd], indices_or_sections=idx_array.shape[channels_axis_2use], axis=channels_axis_2use)]
                # print("roi_mask after channel axis split: ", len(roi_mask_array_2use_2), roi_mask_array_2use_2[0].shape)
            else:
                roi_mask_array_2use_2 = roi_mask_array_2use_1 #which should be meaning None

            #Also split on the channel axis the thresholds' arrays corresponding to the ixd-th index along the analysis_axis
            ch_bin_thresh_2use_2 = split_thresholds_arrays(ch_bin_thresh_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            transform_to_label_img_2use_2 = split_thresholds_arrays(transform_to_label_img_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            val_4zero_regionprops_2use_2 = split_thresholds_arrays(val_4zero_regionprops_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            threshold_roi_mask_2use_2 = split_thresholds_arrays(threshold_roi_mask_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            n_of_region_4areas_measure_2use_2 = split_thresholds_arrays(n_of_region_4areas_measure_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            reg_eucl_dist_within_arr_val_n_regions_nopass_2use_2 = split_thresholds_arrays(reg_eucl_dist_within_arr_val_n_regions_nopass_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            min_px_over_thresh_common_2use_2 = split_thresholds_arrays(min_px_over_thresh_common_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            measure_pixels_overlap_n_px_thr_1_2use_2 = split_thresholds_arrays(measure_pixels_overlap_n_px_thr_1_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            measure_pixels_overlap_n_px_thr_2_2use_2 = split_thresholds_arrays(measure_pixels_overlap_n_px_thr_2_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            count_n_overl_reg_intersection_threshold_2use_2 = split_thresholds_arrays(count_n_overl_reg_intersection_threshold_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=True)
            conv_hull_fract_px_thre_arr_1_2use_2 = split_thresholds_arrays(conv_hull_fract_px_thre_arr_1_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            conv_hull_fract_px_thre_arr_2_2use_2 = split_thresholds_arrays(conv_hull_fract_px_thre_arr_2_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)

            #================================================
            #=========  ITERATE ON THE CHANNEL AXIS =========
            # Iterate through the channels
            for ch_n, ch_array in enumerate(ch_arrays_list):
                # print("---", ch_n)

                #==================================================
                #=========  GET ROI IN THE CORRECT FORMAT =========
                #Get ch_n roi_mask, if it is provided
                if hasattr(roi_mask_array, "__len__"):
                    ch_n_roi_mask_array = roi_mask_array_2use_2[ch_n]
                    # print("final shape roi mask", ch_n_roi_mask_array.shape)
                else:
                    ch_n_roi_mask_array= roi_mask_array_2use_2 #which should be meaning None

                #=================================
                #=========  ANALYSE AREA =========
                #Get threshold value for channel ch_n and index ixd in the analysis axis
                ch_n_ixd_binarization_threshold = get_threshold_from_list(ch_bin_thresh_2use_2[ch_n],
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)
                #Get threshold value for channel ch_n and index ixd in the analysis axis
                ch_n_ixd_value_4_zero_regionprops = get_threshold_from_list(val_4zero_regionprops_2use_2[ch_n],
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)
                
                ch_n_area_px, ch_n_area_props = get_mask_area(ch_array,
                                                              roi_mas_k=ch_n_roi_mask_array,
                                                              binarization_threshold=ch_n_ixd_binarization_threshold,
                                                              value_4_zero_regionprops=ch_n_ixd_value_4_zero_regionprops)
                
                #============================================
                #========= UPDATE OUTPUT DICTIONARY =========
                #Update measurements_dict, which will be used to form the output dataframe
                # modify_dictionary(result_valu_e=ch_n_area_px, dict2modify=measurements_dict, root_key_name='area', channel_1_number=ch_n, channel_2_number=None)

                #==================================
                #=========  COUNT REGIONS =========
                #Count region number
                #Get threshold value for channel ch_n and index ixd in the analysis axis
                ch_n_ixd_threshold_roi_mask = get_threshold_from_list(threshold_roi_mask_2use_2[ch_n],
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)
                
                ch_n_regions_number = count_regions_number(ch_array,
                                                           roi_mask=ch_n_roi_mask_array,
                                                           threshold_input_arr=ch_n_ixd_binarization_threshold,
                                                           threshold_roi_mask=ch_n_ixd_threshold_roi_mask)
                # print("n of regions ", ch_n_regions_number)
                #============================================
                #========= UPDATE OUTPUT DICTIONARY =========
                #Update measurements_dict, which will be used to form the output dataframe
                # modify_dictionary(result_valu_e=ch_n_regions_number, dict2modify=measurements_dict, root_key_name='region_number', channel_1_number=ch_n, channel_2_number=None)

                #===========================================
                #=========  MEASURE REGIONS' AREAS =========
                #Calculate properties of regions, if there are >n_of_region_4areas_measure_2use regions. Alternatively,
                # link properties variables to NaN values
                #Get threshold value for channel ch_n and index ixd in the analysis axis
                ch_n_ixd_n_of_region_4areas_measure = get_threshold_from_list(n_of_region_4areas_measure_2use_2[ch_n],
                                                                                multi_value_array=False,
                                                                                multi_value_axis=-1,
                                                                                get_a_single_value=True)
                #Get threshold value for channel ch_n and cchh_nn at index ixd in the analysis axis
                ch_n_ixd_transform_to_label_img = get_threshold_from_list(transform_to_label_img_2use_2[ch_n],
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)

                if ch_n_regions_number>ch_n_ixd_n_of_region_4areas_measure:
                    #Get the areas of the regions within the channel
                    ch_n_regions_areas = get_areas_of_regions_in_mask(ch_array,
                                                                        roi__mask=ch_n_roi_mask_array,
                                                                        transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                        binarization_threshold=ch_n_ixd_binarization_threshold)
                    
                    #Get mean, median, stdv, sem, max and min regions' area. Get NaN values if a minimum number of areas is not detected
                    ch_n_regions_areas_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_areas, no_quantification_value=no_quantification_valu_e)
                    ch_n_regions_mean_area = ch_n_regions_areas_results[1]
                    ch_n_regions_median_area = ch_n_regions_areas_results[2]
                    ch_n_regions_stdv_area = ch_n_regions_areas_results[3]
                    ch_n_regions_sem_area = ch_n_regions_areas_results[4]
                    ch_n_regions_min_area = ch_n_regions_areas_results[5]
                    ch_n_regions_max_area = ch_n_regions_areas_results[6]
                else:
                    ch_n_regions_mean_area = np.nan
                    ch_n_regions_median_area = np.nan
                    ch_n_regions_stdv_area = np.nan
                    ch_n_regions_sem_area = np.nan
                    ch_n_regions_min_area = np.nan
                    ch_n_regions_max_area = np.nan
                
                #============================================
                #========= UPDATE OUTPUT DICTIONARY =========
                #Update measurements_dict, which will be used to form the output dataframe
                # modify_dictionary(result_valu_e=ch_n_regions_mean_area, dict2modify=measurements_dict, root_key_name='mean_regions_area', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=ch_n_regions_median_area, dict2modify=measurements_dict, root_key_name='median_regions_area', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=ch_n_regions_stdv_area, dict2modify=measurements_dict, root_key_name='stdv_regions_area', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=ch_n_regions_sem_area, dict2modify=measurements_dict, root_key_name='sem_regions_area', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=ch_n_regions_min_area, dict2modify=measurements_dict, root_key_name='min_regions_area', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=ch_n_regions_max_area, dict2modify=measurements_dict, root_key_name='max_regions_area', channel_1_number=ch_n, channel_2_number=None)

                # #=======================================================================
                # #=========  MEASURE INTER-REGIONS DISTANCES WITHIN THE CHANNEL =========
                #Get threshold value for channel ch_n and index ixd in the analysis axis
                ch_n_ixd_highpass_n_regions = get_threshold_from_list(reg_eucl_dist_within_arr_val_n_regions_nopass_2use_2[ch_n],
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)

                ch_n_regions_min_distances, ch_n_regions_min_dict = measure_regions_euclidean_distances_within_array(ch_array,
                                                                                                                        roi__mask=ch_n_roi_mask_array,
                                                                                                                        desired__distance='min',
                                                                                                                        highpass_n_regions=ch_n_ixd_highpass_n_regions,
                                                                                                                        transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                                        label_img_thres=ch_n_ixd_binarization_threshold,
                                                                                                                        return_excluded_distances=False,
                                                                                                                        val_n_regions_nopass=np.nan)
                # ch_n_regions_min_distances is either a list or val_n_regions_nopass value. If it is a list it is because more than highpass_n_regions are present
                # If it is a list, measure the mean and standard deviation if at least 3 distances are present. Only measure the mean if two distances are present.
                # Report the single distance if only 1 distance is present.
                if isinstance(ch_n_regions_min_distances,list):
                    ch_n_regions_min_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_min_distances, no_quantification_value=no_quantification_valu_e)
                    num_ch_n_regions_min_distances = ch_n_regions_min_distances_results[0]
                    mean_ch_n_regions_min_distances = ch_n_regions_min_distances_results[1]
                    median_ch_n_regions_min_distances = ch_n_regions_min_distances_results[2]
                    std_ch_n_regions_min_distances = ch_n_regions_min_distances_results[3]
                    sem_ch_n_regions_min_distances = ch_n_regions_min_distances_results[4]
                    min_ch_n_regions_min_distances = ch_n_regions_min_distances_results[5]
                    max_ch_n_regions_min_distances = ch_n_regions_min_distances_results[6]

                else:
                    num_ch_n_regions_min_distances = np.nan
                    mean_ch_n_regions_min_distances = np.nan
                    median_ch_n_regions_min_distances = np.nan
                    std_ch_n_regions_min_distances = np.nan
                    sem_ch_n_regions_min_distances = np.nan
                    min_ch_n_regions_min_distances = np.nan
                    max_ch_n_regions_min_distances = np.nan
                
                ch_n_regions_max_distances, ch_n_regions_max_dict = measure_regions_euclidean_distances_within_array(ch_array,
                                                                                                                        roi__mask=ch_n_roi_mask_array,
                                                                                                                        desired__distance='max',
                                                                                                                        highpass_n_regions=ch_n_ixd_highpass_n_regions,
                                                                                                                        transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                                        label_img_thres=ch_n_ixd_binarization_threshold,
                                                                                                                        return_excluded_distances=False,
                                                                                                                        val_n_regions_nopass=np.nan)
                
                # ch_n_regions_max_distances is either a list or val_n_regions_nopass value. If it is a list it is because more than highpass_n_regions are present
                # If it is a list, measure the mean and standard deviation if at least 3 distances are present. Only measure the mean if two distances are present.
                # Report the single distance if only 1 distance is present.
                if isinstance(ch_n_regions_max_distances,list):
                    ch_n_regions_max_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_max_distances, no_quantification_value=no_quantification_valu_e)
                    num_ch_n_regions_max_distances = ch_n_regions_max_distances_results[0]
                    mean_ch_n_regions_max_distances = ch_n_regions_max_distances_results[1]
                    median_ch_n_regions_max_distances = ch_n_regions_max_distances_results[2]
                    std_ch_n_regions_max_distances = ch_n_regions_max_distances_results[3]
                    sem_ch_n_regions_max_distances = ch_n_regions_max_distances_results[4]
                    min_ch_n_regions_max_distances = ch_n_regions_max_distances_results[5]
                    max_ch_n_regions_max_distances = ch_n_regions_max_distances_results[6]
                else:
                    num_ch_n_regions_max_distances = np.nan
                    mean_ch_n_regions_max_distances = np.nan
                    median_ch_n_regions_max_distances = np.nan
                    std_ch_n_regions_max_distances = np.nan
                    sem_ch_n_regions_max_distances = np.nan
                    min_ch_n_regions_max_distances = np.nan
                    max_ch_n_regions_max_distances = np.nan

                ch_n_regions_mean_distances, ch_n_regions_mean_dict = measure_regions_euclidean_distances_within_array(ch_array,
                                                                                                                        roi__mask=ch_n_roi_mask_array,
                                                                                                                        desired__distance='mean',
                                                                                                                        highpass_n_regions=ch_n_ixd_highpass_n_regions,
                                                                                                                        transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                                        label_img_thres=ch_n_ixd_binarization_threshold,
                                                                                                                        return_excluded_distances=False,
                                                                                                                        val_n_regions_nopass=np.nan)
                
                # ch_n_regions_mean_distances is either a list or val_n_regions_nopass value. If it is a list it is because more than highpass_n_regions are present
                # If it is a list, measure the mean and standard deviation if at least 3 distances are present. Only measure the mean if two distances are present.
                # Report the single distance if only 1 distance is present.
                if isinstance(ch_n_regions_mean_distances,list):
                    ch_n_regions_mean_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n_regions_mean_distances, no_quantification_value=no_quantification_valu_e)
                    num_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[0]
                    mean_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[1]
                    median_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[2]
                    std_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[3]
                    sem_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[4]
                    min_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[5]
                    max_ch_n_regions_mean_distances = ch_n_regions_mean_distances_results[6]
                else:
                    num_ch_n_regions_mean_distances = np.nan
                    mean_ch_n_regions_mean_distances = np.nan
                    median_ch_n_regions_mean_distances = np.nan
                    std_ch_n_regions_mean_distances = np.nan
                    sem_ch_n_regions_mean_distances = np.nan
                    min_ch_n_regions_mean_distances = np.nan
                    max_ch_n_regions_mean_distances = np.nan

                #============================================
                #========= UPDATE OUTPUT DICTIONARY =========
                #Update measurements_dict, which will be used to form the output dataframe
                # modify_dictionary(result_valu_e=num_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='number_region_min_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=mean_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='mean_region_min_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=median_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='median_region_min_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=std_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='stdv_region_min_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=sem_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='sem_region_min_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=min_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='min_region_min_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=max_ch_n_regions_min_distances, dict2modify=measurements_dict, root_key_name='max_region_min_distances', channel_1_number=ch_n, channel_2_number=None)

                # modify_dictionary(result_valu_e=num_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='number_region_max_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=mean_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='mean_region_max_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=median_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='median_region_max_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=std_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='stdv_region_max_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=sem_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='sem_region_max_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=min_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='min_region_max_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=max_ch_n_regions_max_distances, dict2modify=measurements_dict, root_key_name='max_region_max_distances', channel_1_number=ch_n, channel_2_number=None)

                # modify_dictionary(result_valu_e=num_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='number_region_mean_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=mean_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='mean_region_mean_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=median_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='median_region_mean_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=std_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='stdv_region_mean_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=sem_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='sem_region_mean_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=min_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='min_region_mean_distances', channel_1_number=ch_n, channel_2_number=None)
                # modify_dictionary(result_valu_e=max_ch_n_regions_mean_distances, dict2modify=measurements_dict, root_key_name='max_region_mean_distances', channel_1_number=ch_n, channel_2_number=None)

                #==================================================================
                #=========  START COMPARATIVE MEASUREMENTS AMONG CHANNELS =========
                #Iterate trough the channels a second time, to get measurements calculated by comparing two channels
                for cchh_nn, cchh_nn_array in enumerate(ch_arrays_list):

                    #Avoid measuring a channel angainst itself
                    if ch_n != cchh_nn:
                        # print("...")
                        #===============================================================
                        #=========  VERIFY IF COMPARATIVE ANALYSIS CAN BE DONE =========
                        #Get cchh_nn roi mask if it is provided
                        if hasattr(roi_mask_array, "__len__"):
                            cchh_nn_roi_mask_array = roi_mask_array_2use_2[cchh_nn]
                            # print("final shape roi mask", cchh_nn_roi_mask_array.shape)
                        else:
                            cchh_nn_roi_mask_array= roi_mask_array_2use_2 #which should be meaning None
                        
                        #Count the number of pixels in the second channel
                        #Get threshold value for channel cchh_nn and index ixd in the analysis axis
                        cchh_nn_ixd_binarization_threshold = get_threshold_from_list(ch_bin_thresh_2use_2[cchh_nn],
                                                                                        multi_value_array=False,
                                                                                        multi_value_axis=-1,
                                                                                        get_a_single_value=True)
                        #Get threshold value for channel cchh_nn and index ixd in the analysis axis
                        cchh_nn_ixd_value_4_zero_regionprops = get_threshold_from_list(val_4zero_regionprops_2use_2[cchh_nn],
                                                                                        multi_value_array=False,
                                                                                        multi_value_axis=-1,
                                                                                        get_a_single_value=True)
                        cchh_nn_area_px, cchh_nn_area_props = get_mask_area(cchh_nn_array,
                                                                            roi_mas_k=cchh_nn_roi_mask_array,
                                                                            binarization_threshold=cchh_nn_ixd_binarization_threshold,
                                                                            value_4_zero_regionprops=cchh_nn_ixd_value_4_zero_regionprops)
                        
                        #Do the following analyses only if both channels pass a highpass threshold of number of pixels
                        #Get threshold value for channels ch_n and cchh_nn at index ixd in the analysis axis
                        ch_n_ixd_min_px_of_inter_n = get_threshold_from_list(min_px_over_thresh_common_2use_2[ch_n],
                                                                                    multi_value_array=False,
                                                                                    multi_value_axis=-1,
                                                                                    get_a_single_value=True)
                        
                        cchh_nn_ixd_min_px_of_inter_n = get_threshold_from_list(min_px_over_thresh_common_2use_2[cchh_nn],
                                                                                        multi_value_array=False,
                                                                                        multi_value_axis=-1,
                                                                                        get_a_single_value=True)
                        
                        if ch_n_area_px>ch_n_ixd_min_px_of_inter_n and cchh_nn_area_px>cchh_nn_ixd_min_px_of_inter_n:

                            #==============================================
                            #=========  MEASURE CHANNELS' OVERLAP =========
                            #Measure pixels' overlap
                            #Get threshold value for channel ch_n and cchh_nn at index ixd in the analysis axis
                            ch_n_ixd_n_px_thr_1 = get_threshold_from_list(measure_pixels_overlap_n_px_thr_1_2use_2[ch_n],
                                                                                            multi_value_array=False,
                                                                                            multi_value_axis=-1,
                                                                                            get_a_single_value=True)
                            
                            cchh_nn_ixd_n_px_thr_2 = get_threshold_from_list(measure_pixels_overlap_n_px_thr_2_2use_2[cchh_nn],
                                                                                            multi_value_array=False,
                                                                                            multi_value_axis=-1,
                                                                                            get_a_single_value=True)
                            
                            ch_n__cchh_nn_overlap_i = measure_pixels_overlap(ch_array,
                                                                                cchh_nn_array,
                                                                                roi_mask_arr_1=ch_n_roi_mask_array,
                                                                                roi_mask_arr_2_against=cchh_nn_roi_mask_array,
                                                                                shuffle_times=shuffle_times,
                                                                                n_px_thr_1=ch_n_ixd_n_px_thr_1,
                                                                                n_px_thr_2=cchh_nn_ixd_n_px_thr_2,
                                                                                val_threshold_arr_1=ch_n_ixd_binarization_threshold,
                                                                                val_threshold_arr_2=cchh_nn_ixd_binarization_threshold)
                            
                            #The output of measure_pixels_overlap is different depending on the selected parameters (refer to its documentation).
                            # Get the specific outputs
                            if isinstance(ch_n__cchh_nn_overlap_i, tuple):
                                ch_n__cchh_nn_overlap = ch_n__cchh_nn_overlap_i[0]
                                ch_n__cchh_nn_overlap_shuff = ch_n__cchh_nn_overlap_i[1]
                            else:
                                ch_n__cchh_nn_overlap = np.nan
                                ch_n__cchh_nn_overlap_shuff = [np.nan for shf in range(shuffle_times)]
                            
                            #========================================================
                            #=========  MEASURE REGIONS' EUCLIDEAN DISTANCE =========
                            #Calculate relative distances if at least a region is present in channel cchh_nn (individual regions of ch_n are measured for their distance with
                            # regions of cchh_nn)
                            if cchh_nn_area_px>0:
                                ch_n__cchh_nn_min_euclid_distances_list, ch_n__cchh_nn_min_euclid_distances_dict = measure_regions_euclidean_distances(ch_array,
                                                                                                                cchh_nn_array,
                                                                                                                roi__mask_img1=ch_n_roi_mask_array,
                                                                                                                roi__mask_targ=cchh_nn_roi_mask_array,
                                                                                                                desired__distance='min',
                                                                                                                transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                                label_img_1_thres=ch_n_ixd_binarization_threshold,
                                                                                                                binary_mask_target_thres=cchh_nn_ixd_binarization_threshold)
                                
                                #The output of ch_n__cchh_nn_min_euclid_distances_list in the first position is a list. If no region is present in ch_array this list is empty.
                                #Get the mean and standard deviation of regions minimum euclidean distances if there are at least 2 distances. Only measure the mean if there is only 1
                                #distance. Get the min and max minimum euclidean distances. Use np.nan if no distance is present.
                                ch_n__cchh_nn_min_euclid_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n__cchh_nn_min_euclid_distances_list,
                                                                                                                     no_quantification_value=no_quantification_valu_e)
                                mean_ch_n__cchh_nn_min_euclid_distances = ch_n__cchh_nn_min_euclid_distances_results[1]
                                median_ch_n__cchh_nn_min_euclid_distances = ch_n__cchh_nn_min_euclid_distances_results[2]
                                std_ch_n__cchh_nn_min_euclid_distances = ch_n__cchh_nn_min_euclid_distances_results[3]
                                sem_ch_n__cchh_nn_min_euclid_distances = ch_n__cchh_nn_min_euclid_distances_results[4]
                                min_ch_n__cchh_nn_min_euclid_distances = ch_n__cchh_nn_min_euclid_distances_results[5]
                                max_ch_n__cchh_nn_min_euclid_distances = ch_n__cchh_nn_min_euclid_distances_results[6]
                                
                                ch_n__cchh_nn_max_euclid_distances_list, ch_n__cchh_nn_max_euclid_distances_dict = measure_regions_euclidean_distances(ch_array,
                                                                                                                cchh_nn_array,
                                                                                                                roi__mask_img1=ch_n_roi_mask_array,
                                                                                                                roi__mask_targ=cchh_nn_roi_mask_array,
                                                                                                                desired__distance='max',
                                                                                                                transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                                label_img_1_thres=ch_n_ixd_binarization_threshold,
                                                                                                                binary_mask_target_thres=cchh_nn_ixd_binarization_threshold)
                                
                                # The output of ch_n__cchh_nn_max_euclid_distances_list in the first position is a list. If no region is present in ch_array this list is empty.
                                # Get the mean and standard deviation of regions maximum euclidean distances if there are at least 2 distances. Only measure the mean if there is only 1
                                # distance. Get the min and max maximum euclidean distances. Use np.nan if no distance is present.
                                ch_n__cchh_nn_max_euclid_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n__cchh_nn_max_euclid_distances_list,
                                                                                                                     no_quantification_value=no_quantification_valu_e)
                                mean_ch_n__cchh_nn_max_euclid_distances = ch_n__cchh_nn_max_euclid_distances_results[1]
                                median_ch_n__cchh_nn_max_euclid_distances = ch_n__cchh_nn_max_euclid_distances_results[2]
                                std_ch_n__cchh_nn_max_euclid_distances = ch_n__cchh_nn_max_euclid_distances_results[3]
                                sem_ch_n__cchh_nn_max_euclid_distances = ch_n__cchh_nn_max_euclid_distances_results[4]
                                min_ch_n__cchh_nn_max_euclid_distances = ch_n__cchh_nn_max_euclid_distances_results[5]
                                max_ch_n__cchh_nn_max_euclid_distances = ch_n__cchh_nn_max_euclid_distances_results[6]

                                ch_n__cchh_nn_mean_euclid_distance_list, ch_n__cchh_nn_mean_euclid_distance_dict = measure_regions_euclidean_distances(ch_array,
                                                                                                            cchh_nn_array,
                                                                                                            roi__mask_img1=ch_n_roi_mask_array,
                                                                                                            roi__mask_targ=cchh_nn_roi_mask_array,
                                                                                                            desired__distance='mean',
                                                                                                            transform_to_label_img=ch_n_ixd_transform_to_label_img,
                                                                                                            label_img_1_thres=ch_n_ixd_binarization_threshold,
                                                                                                            binary_mask_target_thres=cchh_nn_ixd_binarization_threshold)

                                #The output of ch_n__cchh_nn_mean_euclid_distance in the first position is a list. If no region is present in ch_array this list is empty.
                                #Get the mean and standard deviation of regions mean euclidean distances if there are at least 2 distances. Only measure the mean if there is only 1
                                #distance. Get the min and max mean euclidean distances. Use np.nan if no distance is present.
                                ch_n__cchh_nn_mean_euclid_distances_results = get_mean_median_std_sem_min_max_results(results_measurements=ch_n__cchh_nn_mean_euclid_distance_list,
                                                                                                                     no_quantification_value=no_quantification_valu_e)
                                mean_ch_n__cchh_nn_mean_euclid_distances = ch_n__cchh_nn_mean_euclid_distances_results[1]
                                median_ch_n__cchh_nn_mean_euclid_distances = ch_n__cchh_nn_mean_euclid_distances_results[2]
                                std_ch_n__cchh_nn_mean_euclid_distances = ch_n__cchh_nn_mean_euclid_distances_results[3]
                                sem_ch_n__cchh_nn_mean_euclid_distances = ch_n__cchh_nn_mean_euclid_distances_results[4]
                                min_ch_n__cchh_nn_mean_euclid_distances = ch_n__cchh_nn_mean_euclid_distances_results[5]
                                max_ch_n__cchh_nn_mean_euclid_distances = ch_n__cchh_nn_mean_euclid_distances_results[6]
                                
                            #Assign np.nan values to measurements if no region is present in channel cchh_nn
                            else:
                                mean_ch_n__cchh_nn_min_euclid_distances = np.nan
                                median_ch_n__cchh_nn_min_euclid_distances = np.nan
                                std_ch_n__cchh_nn_min_euclid_distances = np.nan
                                sem_ch_n__cchh_nn_min_euclid_distances = np.nan
                                min_ch_n__cchh_nn_min_euclid_distances = np.nan
                                max_ch_n__cchh_nn_min_euclid_distances = np.nan

                                mean_ch_n__cchh_nn_max_euclid_distances = np.nan
                                median_ch_n__cchh_nn_max_euclid_distances = np.nan
                                std_ch_n__cchh_nn_max_euclid_distances = np.nan
                                sem_ch_n__cchh_nn_max_euclid_distances = np.nan
                                min_ch_n__cchh_nn_max_euclid_distances = np.nan
                                max_ch_n__cchh_nn_max_euclid_distances = np.nan

                                mean_ch_n__cchh_nn_mean_euclid_distances = np.nan
                                median_ch_n__cchh_nn_mean_euclid_distances = np.nan
                                std_ch_n__cchh_nn_mean_euclid_distances = np.nan
                                sem_ch_n__cchh_nn_mean_euclid_distances = np.nan
                                min_ch_n__cchh_nn_mean_euclid_distances = np.nan
                                max_ch_n__cchh_nn_mean_euclid_distances = np.nan
                            
                            #========================================================
                            #=========  COUNT NUMBER OF OVERLAPPING REGIONS =========
                            #Get threshold value for channel ch_n and cchh_nn at index ixd in the analysis axis
                            cchh_nn_ixd_transform_to_label_img = get_threshold_from_list(transform_to_label_img_2use_2[cchh_nn],
                                                                                            multi_value_array=False,
                                                                                            multi_value_axis=-1,
                                                                                            get_a_single_value=True)
                            #Get intersection threshold
                            ch_n_cchh_nn_ixd_intersection_threshold_tuple = get_threshold_from_list(count_n_overl_reg_intersection_threshold_2use_2[ch_n],
                                                                                                multi_value_array=True,
                                                                                                multi_value_axis=-1,
                                                                                                get_a_single_value=True)
                            
                            position_of_intersection_threshold = (ch_n*(channels_array.shape[channels_axis]))+cchh_nn #get the position of the threshold in the multi-threshold tuple
                            ch_n_cchh_nn_ixd_intersection_threshold = ch_n_cchh_nn_ixd_intersection_threshold_tuple[position_of_intersection_threshold]

                            ch_n__cchh_nn_overlapping_regions_i,dum_my,dum_my1,dum_my2 = count_number_of_overlapping_regions(ch_array,
                                                                                                    cchh_nn_array,
                                                                                                    intersection_threshold=ch_n_cchh_nn_ixd_intersection_threshold,
                                                                                                    ro_i__mask_1=ch_n_roi_mask_array,
                                                                                                    ro_i__mask_2=cchh_nn_roi_mask_array,
                                                                                                    transform__to_label_img_arr_1=ch_n_ixd_transform_to_label_img,
                                                                                                    transform__to_label_img_arr_2=cchh_nn_ixd_transform_to_label_img,
                                                                                                    arr_1_tot_thres=ch_n_ixd_binarization_threshold,
                                                                                                    arr_2_part_thres=cchh_nn_ixd_binarization_threshold,
                                                                                                    return_regions=False,
                                                                                                    return_intersection_arrays=False,
                                                                                                    output_arr_loval=0, #only applies if return_intersection_arrays=True
                                                                                                    output_arr_highval=255, #only applies if return_intersection_arrays=True
                                                                                                    output_arr_dtype=np.uint8) #only applies if return_intersection_arrays=True
                            
                            # The output of ch_n__cchh_nn_overlapping_regions in position 0 is a dictionary. If no region is present in ch_n the dictionary is empty.
                            # Link the quantification to np.nan if the dictionary is empty, keep the output otherwise
                            if len(ch_n__cchh_nn_overlapping_regions_i)>0:
                                ch_n__cchh_nn_overlapping_regions=ch_n__cchh_nn_overlapping_regions_i
                            else:
                                ch_n__cchh_nn_overlapping_regions=np.nan

                            #==================================================
                            #=========  MEASURE CONVEX HULL FRACTIONS =========
                            #Get threshold values for channel ch_n and cchh_nn at index ixd in the analysis axis
                            cchh_nn_ixd_threshold_roi_mask = get_threshold_from_list(threshold_roi_mask_2use_2[cchh_nn],
                                                                            multi_value_array=False,
                                                                            multi_value_axis=-1,
                                                                            get_a_single_value=True)
                            
                            ch_n_ixd_px_thre_arr_1 = get_threshold_from_list(conv_hull_fract_px_thre_arr_1_2use_2[ch_n],
                                                                                multi_value_array=False,
                                                                                multi_value_axis=-1,
                                                                                get_a_single_value=True)
                            
                            cchh_nn_ixd_px_thre_arr_2 = get_threshold_from_list(conv_hull_fract_px_thre_arr_2_2use_2[cchh_nn],
                                                                                multi_value_array=False,
                                                                                multi_value_axis=-1,
                                                                                get_a_single_value=True)
                            
                            ch_n__cchh_nn_convex_hull_fraction = get_convex_hull_fraction(ch_array,
                                                                                          cchh_nn_array,
                                                                                          roi__mask_1=ch_n_roi_mask_array,
                                                                                          roi__mask_2=cchh_nn_roi_mask_array,
                                                                                          threshold_arr_1=ch_n_ixd_binarization_threshold,
                                                                                          threshold_arr_2=cchh_nn_ixd_binarization_threshold,
                                                                                          threshold_roi_mask_1=ch_n_ixd_threshold_roi_mask,
                                                                                          threshold_roi_mask_2=cchh_nn_ixd_threshold_roi_mask,
                                                                                          px_thre_arr_1=ch_n_ixd_px_thre_arr_1,
                                                                                          px_thre_arr_2=cchh_nn_ixd_px_thre_arr_2,
                                                                                          val_4_arr1_NOpassthres_arr2_passthres=get_conv_hull_fract_arr1_NOpass_arr2_pass_v,
                                                                                          val_4_arr2_NOpassthres=get_conv_hull_fract_arr2_NOpass_v)
                        
                        #========================================================================================
                        #=========  ADD NaNs VALUES AS RESULTS OF THE ANALYSES IF THEY COULD NOT BE DONE =========
                        else:
                            #channels' overlap
                            ch_n__cchh_nn_overlap = np.nan
                            ch_n__cchh_nn_overlap_shuff = [np.nan for shf in range(shuffle_times)]

                            #inter-channels euclidean distances
                            mean_ch_n__cchh_nn_min_euclid_distances = np.nan
                            median_ch_n__cchh_nn_min_euclid_distances = np.nan
                            std_ch_n__cchh_nn_min_euclid_distances = np.nan
                            sem_ch_n__cchh_nn_min_euclid_distances = np.nan
                            min_ch_n__cchh_nn_min_euclid_distances = np.nan
                            max_ch_n__cchh_nn_min_euclid_distances = np.nan

                            mean_ch_n__cchh_nn_max_euclid_distances = np.nan
                            median_ch_n__cchh_nn_max_euclid_distances = np.nan
                            std_ch_n__cchh_nn_max_euclid_distances = np.nan
                            sem_ch_n__cchh_nn_max_euclid_distances = np.nan
                            min_ch_n__cchh_nn_max_euclid_distances = np.nan
                            max_ch_n__cchh_nn_max_euclid_distances = np.nan

                            mean_ch_n__cchh_nn_mean_euclid_distances = np.nan
                            median_ch_n__cchh_nn_mean_euclid_distances = np.nan
                            std_ch_n__cchh_nn_mean_euclid_distances = np.nan
                            sem_ch_n__cchh_nn_mean_euclid_distances = np.nan
                            min_ch_n__cchh_nn_mean_euclid_distances = np.nan
                            max_ch_n__cchh_nn_mean_euclid_distances = np.nan

                            #overlapping regions
                            ch_n__cchh_nn_overlapping_regions = np.nan

                            #convex hull
                            ch_n__cchh_nn_convex_hull_fraction = np.nan

                        #============================================
                        #========= UPDATE OUTPUT DICTIONARY =========
                        # #Update measurements_dict, which will be used to form the output dataframe

                        # #channels' overlap
                        # modify_dictionary(result_valu_e=ch_n__cchh_nn_overlap, dict2modify=measurements_dict, root_key_name='pixels_overlap_observed', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=ch_n__cchh_nn_overlap_shuff, dict2modify=measurements_dict, root_key_name='pixels_overlap_shuffle', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        
                        # #inter-channels euclidean distances
                        # modify_dictionary(result_valu_e=mean_ch_n__cchh_nn_min_euclid_distances, dict2modify=measurements_dict, root_key_name='mean_min_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=median_ch_n__cchh_nn_min_euclid_distances, dict2modify=measurements_dict, root_key_name='median_min_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=std_ch_n__cchh_nn_min_euclid_distances, dict2modify=measurements_dict, root_key_name='stdv_min_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=sem_ch_n__cchh_nn_min_euclid_distances, dict2modify=measurements_dict, root_key_name='sem_min_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=min_ch_n__cchh_nn_min_euclid_distances, dict2modify=measurements_dict, root_key_name='min_min_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=max_ch_n__cchh_nn_min_euclid_distances, dict2modify=measurements_dict, root_key_name='max_min_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                            
                        # modify_dictionary(result_valu_e=mean_ch_n__cchh_nn_max_euclid_distances, dict2modify=measurements_dict, root_key_name='mean_max_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=median_ch_n__cchh_nn_max_euclid_distances, dict2modify=measurements_dict, root_key_name='median_max_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=std_ch_n__cchh_nn_max_euclid_distances, dict2modify=measurements_dict, root_key_name='stdv_max_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=sem_ch_n__cchh_nn_max_euclid_distances, dict2modify=measurements_dict, root_key_name='sem_max_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=min_ch_n__cchh_nn_max_euclid_distances, dict2modify=measurements_dict, root_key_name='min_max_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=max_ch_n__cchh_nn_max_euclid_distances, dict2modify=measurements_dict, root_key_name='max_max_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)

                        # modify_dictionary(result_valu_e=mean_ch_n__cchh_nn_mean_euclid_distances, dict2modify=measurements_dict, root_key_name='mean_mean_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=median_ch_n__cchh_nn_mean_euclid_distances, dict2modify=measurements_dict, root_key_name='median_mean_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=std_ch_n__cchh_nn_mean_euclid_distances, dict2modify=measurements_dict, root_key_name='stdv_mean_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=sem_ch_n__cchh_nn_mean_euclid_distances, dict2modify=measurements_dict, root_key_name='sem_mean_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=min_ch_n__cchh_nn_mean_euclid_distances, dict2modify=measurements_dict, root_key_name='min_mean_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                        # modify_dictionary(result_valu_e=max_ch_n__cchh_nn_mean_euclid_distances, dict2modify=measurements_dict, root_key_name='max_mean_distance_regions', channel_1_number=ch_n, channel_2_number=cchh_nn)
                                                
                        # #Also update count_number_of_overlapping_regions_coll_dict, to be used for adding the measurements of count_number_of_overlapping_regions to measurements_dict,
                        # #at the end of the iterations
                        count_number_of_overlapping_regions_coll_dict[ixd][(ch_n,cchh_nn)]=ch_n__cchh_nn_overlapping_regions

                        #convex hull
                        # modify_dictionary(result_valu_e=ch_n__cchh_nn_overlap, dict2modify=measurements_dict, root_key_name='pixels_overlap_observed', channel_1_number=ch_n, channel_2_number=cchh_nn)

        
        #====================================================================================================
        #========= UPDATE OUTPUT DICTIONARY FOR THE RESULTS OF  count_number_of_overlapping_regions =========
        #Get the higher number of counted overlapping regions
        counted_overlapping_region_coll_list = [] #Initialize a collection list
        #Iterate through the indexes along analysis axis within count_number_of_overlapping_regions_coll_dict
        for ixd_1 in count_number_of_overlapping_regions_coll_dict:
            #Iterate through the channels couple at each index ixd_1
            for ch_coupl in count_number_of_overlapping_regions_coll_dict[ixd_1]:
                #Iterate through the counted overlapping regions, if the channel_couple ch_coupl at index ixd_1 could be quantified
                if isinstance(count_number_of_overlapping_regions_coll_dict[ixd_1][ch_coupl], dict):
                    for over_count in count_number_of_overlapping_regions_coll_dict[ixd_1][ch_coupl]:
                        counted_overlapping_region_coll_list.append(over_count)
        #Remove duplicate counts from counted_overlapping_region_coll_list and sort number ascending
        sorted_unique_counted_overlapping_region_coll_list = sorted(list(set(counted_overlapping_region_coll_list)))
        #Iterate through the sorted counted number of overlapping regions
        for scor in sorted_unique_counted_overlapping_region_coll_list:
            #Iterate through the indexes along analysis axis within count_number_of_overlapping_regions_coll_dict
            for ixd_2 in count_number_of_overlapping_regions_coll_dict:
                #Iterate through the channels couple at each index ixd_2
                for ch_coupl_1 in count_number_of_overlapping_regions_coll_dict[ixd_2]:
                    #form the column name
                    column_name_4reg_overlap_measure = f"n_ch_{ch_coupl_1[0]}_regions_overlap_w_{scor}_ch_{ch_coupl_1[1]}_regions"
                    #Check if counted number of overlapping regions has been quantified for index ixd_2 and channel couple ch_coupl_1
                    try:
                        #If they have been quantified, add them to the output dictionary
                        quantification_result = count_number_of_overlapping_regions_coll_dict[ixd_2][ch_coupl_1][scor]
                        modify_dictionary(result_valu_e=quantification_result,
                                          dict2modify=measurements_dict,
                                          root_key_name=column_name_4reg_overlap_measure,
                                          channel_1_number=None,
                                          channel_2_number=None)
                    except:
                        #If the counted number of overlapping regions are not present, but the channel couple ch_coupl_1 at index ixd_2 was quantified, it means that there are
                        #no regions in the first channels which overlap with regions of the second channel the counted number of times. Add 0 to the collected dictionary
                        if isinstance(count_number_of_overlapping_regions_coll_dict[ixd_2][ch_coupl_1], dict):
                            modify_dictionary(result_valu_e=0.0,
                                            dict2modify=measurements_dict,
                                            root_key_name=column_name_4reg_overlap_measure,
                                            channel_1_number=None,
                                            channel_2_number=None)
                        #If no dictionary has been associated to the channel couple ch_coupl_1 at index ixd_2, it means that no quantification could be done.
                        #Add np.nan to the output dictionary
                        else:
                            modify_dictionary(result_valu_e=count_number_of_overlapping_regions_coll_dict[ixd_2][ch_coupl_1],
                                            dict2modify=measurements_dict,
                                            root_key_name=column_name_4reg_overlap_measure,
                                            channel_1_number=None,
                                            channel_2_number=None)


    # # #If the analysis axis is not provided          
    # # else:
    # #     print("==="*10)
    # #     print("---NO analysis of a specific axis---")
    # #     print("==="*10)

    # #     # Iterate through the channels
    # #     for ch_n1, ch_array1 in enumerate([np.squeeze(c) for c in np.split(channels_array_copy,
    # #                                                                          indices_or_sections=channels_array_copy.shape[channels_axis_2use],
    # #                                                                          axis=channels_axis_2use)]):
    # #         if hasattr(roi_mask_array, "__len__"):
    # #             print("shape roi mask", roi_mask_array_copy.shape)
    # #         else:
    # #             print(roi_mask_array_copy)
    # return
