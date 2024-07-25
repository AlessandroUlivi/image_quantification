import numpy as np
import pandas as pd
from co_localization_measurements import measure_pixels_overlap, measure_regions_euclidean_distances, count_number_of_overlapping_regions
from counting_measurements import count_regions_number
from geometric_measurements import get_mask_area, get_areas_of_regions_in_mask
from topological_measurement import get_convex_hull_fraction
from utils import match_arrays_dimensions


# measure_pixels_overlap,
# measure_regions_euclidean_distances,
# count_number_of_overlapping_regions
# get_convex_hull_fractions


def quantify_channels(channels_array, channels_axis=0, roi_mask_array=None, analysis_axis=None, shuffle_times=0,
                      channels_binarization_thresholds=0, get_mask_area_val_4zero_regionprops=0, count_regions_number_threshold_roi_mask=0, n_of_region_4areas_measure=0,
                      min_px_over_thresh_common=-1, measure_pixels_overlap_n_px_thr_1=1, measure_pixels_overlap_n_px_thr_2=0, reg_eucl_dist_transform_to_label_img=False):
    """
    seems to work with 3 limitations: 1) when providing the thresholds they array cannot contain axis of size 1. 2) when channel_axis/analysis_axis is in position 0 it can't
    be indicated using negative number indexing (for example -10). 3) If a custom array of thresholds is provided and multiple thresholds are required (e.g. for comparison
    functions, the multiple thresholds must be in position -1)
    - roi_mask_array can be different for the channels. At least 1 axis mutch match channels_arrays. The matching axis must be in the correct position.
    - min_px_over_thresh_common. the number of o pixels both channels must pass to continue with paired measurements.
    NOTE than when a tuple or a list is passed as a threshold this is interpreted as a multi-threshold, not as individual thresholds for the different channels.
    NOTE WELL THAT reg_eucl_dist_transform_to_label_img=False.
    """
    #=========================================
    #=========  SUPPORTING FUNCTIONS =========
    def modify_dictionary(dict2modify, key_name, valu_e=None, interations_times=1):
        for i in range(interations_times):
            if interations_times==1:
                key_name_1 = key_name
            else:
                key_name_1 = key_name + "_"+str(i)
            
            if key_name_1 not in dict2modify:
                if valu_e == None:
                    dict2modify[key_name_1]=[]
                else:
                    dict2modify[key_name_1]=[valu_e]
            else:
                if valu_e != None:
                    dict2modify[key_name_1].append(valu_e)
                else:
                    print("WARNING! ", key_name_1, "'s value is updated by no value is provided for the update. None will be used instead")
                    dict2modify[key_name_1].append(valu_e)
    
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
        # print("initial_roi_mask_shape: ", roi_mask_array_2use.shape)
    else:
        roi_mask_array_2use = roi_mask_array

    #=========================================================
    #=========  GET THRESHOLDS IN THE CORRECT FORMAT =========
    #Set binarization thresholds to 0 for all channels, if channels_binarization_thresholds is not provided. Use provided values othewise.
    ch_bin_thresh_2use = set_thresholds_2use(channels_binarization_thresholds, channels_stac_k=channels_array_copy)
    
    #Set val_4zero_regionprops in get_mask_area as 0 by defaut, if None is provided as input. Use the provided value otherwise.
    val_4zero_regionprops_2use = set_thresholds_2use(get_mask_area_val_4zero_regionprops, channels_stac_k=channels_array_copy)

    #Set threshold_roi_mask in count_regions_number as 0 by defaut, if None is provided as input. Use the provided value otherwise.
    threshold_roi_mask_2use = set_thresholds_2use(count_regions_number_threshold_roi_mask, channels_stac_k=channels_array_copy)

    #Set to 0 the highpass threshold for calculating the mean, median, min and max area of regions within a channel, if None is provided as input. Use the provided value otherwise.
    n_of_region_4areas_measure_2use = set_thresholds_2use(n_of_region_4areas_measure, channels_stac_k=channels_array_copy)

    #Set to 0 the min number of pixels for proceeding with measurements, if min_px_over_thresh_common is not provided. Use the provided thresholds otherwise
    min_px_over_thresh_common_2use = set_thresholds_2use(min_px_over_thresh_common, channels_stac_k=channels_array_copy)
    
    #Set to 1 the min number of pixels of arr_1 for calculating pixels overlap in measure_pixels_overlap, if measure_pixels_overlap_n_px_thr_1 is not provided.
    # Use the provided thresholds otherwise
    measure_pixels_overlap_n_px_thr_1_2use = set_thresholds_2use(measure_pixels_overlap_n_px_thr_1, channels_stac_k=channels_array_copy)
    
    #Set to 0 the min number of pixels of arr_2_against for calculating pixels overlap in measure_pixels_overlap, if measure_pixels_overlap_n_px_thr_2 is not provided.
    # Use the provided thresholds otherwise
    measure_pixels_overlap_n_px_thr_2_2use = set_thresholds_2use(measure_pixels_overlap_n_px_thr_2, channels_stac_k=channels_array_copy)

    #==========================================
    #=========  INITIALIZE THE OUTPUT =========
    #Initialize a dictionary to be used to be used to form the output datafram
    measurements_dict = {}

    #==================================================
    #=========  USE ANALYSIS AXIS IF PROVIDED =========
    #If analysis axis is provided:
    if analysis_axis != None:
        # print("==="*10)
        # print("---analyze a specific axis---")
        # print("==="*10)
        
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
        val_4zero_regionprops_2use_1 = split_thresholds_arrays(val_4zero_regionprops_2use, split_axis=analysis_axis, multi_thresholds=False)
        threshold_roi_mask_2use_1 = split_thresholds_arrays(threshold_roi_mask_2use, split_axis=analysis_axis, multi_thresholds=False)
        n_of_region_4areas_measure_2use_1 = split_thresholds_arrays(n_of_region_4areas_measure_2use, split_axis=analysis_axis, multi_thresholds=False)
        min_px_over_thresh_common_2use_1 = split_thresholds_arrays(min_px_over_thresh_common_2use, split_axis=analysis_axis, multi_thresholds=False)
        measure_pixels_overlap_n_px_thr_1_2use_1 = split_thresholds_arrays(measure_pixels_overlap_n_px_thr_1_2use, split_axis=analysis_axis, multi_thresholds=False)
        measure_pixels_overlap_n_px_thr_2_2use_1 = split_thresholds_arrays(measure_pixels_overlap_n_px_thr_2_2use, split_axis=analysis_axis, multi_thresholds=False)

        #=================================================
        #=========  ITERATE ON THE ANALYSIS AXIS =========
        # Iterate through the analysis axis
        for ixd, idx_array in enumerate([np.squeeze(a) for a in np.split(channels_array_copy,
                                                                         indices_or_sections=channels_array_copy.shape[analysis_axis],
                                                                         axis=analysis_axis)]):
            # print("===", ixd)
            #===================================================================================
            #=========  PREPARE DATA, ROI AND THRESHOLDS FOR ITERATION ON CHANNEL AXIS =========
            #Get the individual channels array as a list
            ch_arrays_list = [np.squeeze(b) for b in np.split(idx_array, indices_or_sections=idx_array.shape[channels_axis_2use], axis=channels_axis_2use)]

            #Also che the individual channels arrays as a list for the roi_mask, if it is provided
            if hasattr(roi_mask_array, "__len__"):
                roi_mask_array_2use_2 = [np.squeeze(v) for v in np.split(roi_mask_array_2use_1[ixd], indices_or_sections=idx_array.shape[channels_axis_2use], axis=channels_axis_2use)]
                # print("roi_mask after channel axis split: ", len(roi_mask_array_2use_2), roi_mask_array_2use_2[0].shape)
            else:
                roi_mask_array_2use_2 = roi_mask_array_2use_1 #which should be meaning None

            #Also split on the channel axis the thresholds' arrays corresponding to the ixd-th index along the analysis_axis
            ch_bin_thresh_2use_2 = split_thresholds_arrays(ch_bin_thresh_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            val_4zero_regionprops_2use_2 = split_thresholds_arrays(val_4zero_regionprops_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            threshold_roi_mask_2use_2 = split_thresholds_arrays(threshold_roi_mask_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            n_of_region_4areas_measure_2use_2 = split_thresholds_arrays(n_of_region_4areas_measure_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            min_px_over_thresh_common_2use_2 = split_thresholds_arrays(min_px_over_thresh_common_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            measure_pixels_overlap_n_px_thr_1_2use_2 = split_thresholds_arrays(measure_pixels_overlap_n_px_thr_1_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)
            measure_pixels_overlap_n_px_thr_2_2use_2 = split_thresholds_arrays(measure_pixels_overlap_n_px_thr_2_2use_1[ixd], split_axis=channels_axis_2use, multi_thresholds=False)

            #================================================
            #=========  ITERATE ON THE CHANNEL AXIS =========
            # Iterate through the channels
            for ch_n, ch_array in enumerate(ch_arrays_list):
                # print("===", ch_n)
                #Get ch_n roi_mask, if it is provided

                #==================================================
                #=========  GET ROI IN THE CORRECT FORMAT =========
                if hasattr(roi_mask_array, "__len__"):
                    ch_n_roi_mask_array = roi_mask_array_2use_2[ch_n]
                    # print("final shape roi mask", ch_n_roi_mask_array.shape)
                else:
                    ch_n_roi_mask_array= roi_mask_array_2use_2 #which should be meaning None

                #=================================
                #=========  ANALYSE AREA =========
                #Get mask area
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
                
                #===========================================
                #=========  MEASURE REGIONS' AREAS =========
                #Calculate mean, median, max and min regions' area, if there are >n_of_region_4areas_measure_2use regions. Alternatively,
                # link mean, median, max and min variables to NaN values
                #Get threshold value for channel ch_n and index ixd in the analysis axis
                ch_n_ixd_n_of_region_4areas_measure = get_threshold_from_list(n_of_region_4areas_measure_2use_2[ch_n],
                                                                                multi_value_array=False,
                                                                                multi_value_axis=-1,
                                                                                get_a_single_value=True)
                if ch_n_regions_number>ch_n_ixd_n_of_region_4areas_measure:
                    #Get the areas of the regions within the channel
                    ch_n_regions_areas = get_areas_of_regions_in_mask(ch_array,
                                                                    roi__mask=ch_n_roi_mask_array,
                                                                    transform_to_label_img=True,
                                                                    binarization_threshold=ch_n_ixd_binarization_threshold)
                    
                    #Get mean, median, max and min regions' area. Get NaN values if a minimum number of areas is not detected
                    ch_n_regions_mean_area = np.mean(ch_n_regions_areas)
                    ch_n_regions_median_area = np.median(ch_n_regions_areas)
                    ch_n_regions_max_area = np.amax(ch_n_regions_areas)
                    ch_n_regions_min_area = np.amin(ch_n_regions_areas)
                else:
                    ch_n_regions_mean_area = np.nan
                    ch_n_regions_median_area = np.nan
                    ch_n_regions_max_area = np.nan
                    ch_n_regions_min_area = np.nan

                #==================================================================
                #=========  START COMPARATIVE MEASUREMENTS AMONG CHANNELS =========
                #Iterate trough the channels a second time, to get measurements calculated by comparing two channels
                for cchh_nn, cchh_nn_array in enumerate(ch_arrays_list):
                    
                    #Avoid measuring a channel angainst itself
                    if ch_n != cchh_nn:
                        
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
                        #Get threshold value for channel ch_n and cchh_nn at index ixd in the analysis axis
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
                            # ch_n__cchh_nn_min_euclid_distances = measure_regions_euclidean_distances(ch_array,
                            #                                                                             cchh_nn_array,
                            #                                                                             roi__mask_img1=ch_n_roi_mask_array,
                            #                                                                             roi__mask_targ=cchh_nn_roi_mask_array,
                            #                                                                             desired__distance='min',
                            #                                                                             transform_to_label_img=,
                            #                                                                             label_img_1_thres=ch_n_ixd_binarization_threshold,
                            #                                                                             binary_mask_target_thres=cchh_nn_ixd_binarization_threshold)

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
