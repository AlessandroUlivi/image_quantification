import os
import tifffile
import numpy as np
from random import sample
import pandas as pd
from utils import listdirNHF
from scipy.stats import spearmanr
from image_filtering import blur_image


def analyse_spearman_array(arr_1, arr_2, roi_mask=None, smooth_image_1=False, gaus_n_1=3, gaus_ny_1=None):
    """
    returns the spearman correlation coeffient and p-value of two arrays. If a roi_mask is provided, only consider the pixels within the roi_mask (positive pixels in the roi_mask
    are assumed to be the pixels of interest).
    If smooth_image is set to True, input arrays will be smoothing using a gaussian kernel of size 3 pixels before calculating spearman correlation coefficient and p-value.
    The present function has been tested on 2D arrays.
    """
    #Copy input arrays
    arr_1_copy = arr_1.copy()
    arr_2_copy = arr_2.copy()

    #Gaussian smooth input arrays if smooth_image is set to True
    if smooth_image_1:
        arr_1_to_proc = blur_image(arr_1_copy, n=gaus_n_1, ny=gaus_ny_1)
        arr_2_to_proc = blur_image(arr_2_copy, n=gaus_n_1, ny=gaus_ny_1)
    else:
        arr_1_to_proc = arr_1_copy
        arr_2_to_proc = arr_2_copy

    #If a roi is indicated, restrict the analysis to the roi
    if hasattr(roi_mask, "__len__"):
        #Copy roi_mask
        roi_mask_copy = roi_mask.copy

        #Select the roi pixels
        roi_arr_1 = arr_1_to_proc[roi_mask>0]
        roi_arr_2 = arr_2_to_proc[roi_mask>0]
    
    else:
        #Initialize a dummy ones array
        dummy_zero = np.ones(arr_1.shape)
        #Use the dummy array to select the whole image. Note: the present step flattens the array
        roi_arr_1 = arr_1_to_proc[dummy_zero>0]
        roi_arr_2 = arr_2_to_proc[dummy_zero>0]

    #Calculate spearmann correlation coefficients and p-values
    coeff, p_val = spearmanr(roi_arr_1, roi_arr_2)

    return coeff, p_val


def analyse_spearman_3Darray(arr_1, arr_2, shuffle_times=5, smooth_image=False, gaus_n=3, roi_mask=None, pixels_threshold=500, axis_to_use=0, return_dict=False):
    """
    Given a two 3D arrays, calculates the spearman correlation coefficient and p-value of 2D arrays along a given axis (defaul is axis 0).
    Per each 2D array, the spearman correlation coefficients and p-values are also calculated n times (n=shuffle_times, default is 100) for arrays obtained by randomly shuffling
    the pixels.
    If a binary roi_mask is provided, the analysis is restricted to the pixels of the roi (positive pixels in the binary mask are assumed to be the pixels of interest).
    If a binary roi_mask is provided, the analysis is conducted only when the number of positive pixels in the roi_mask is higher than pixels_threshold.
    The defauls output is a pandas DataFrame with the observed quantifications, the quantifications of each individual shuffle, and the mean and standard deviation of the shuffle.
    If return_dict is selected, the dictionary collecting the quantifications and used to create the output pandas dataframe is also returned.
    """
    #Copy arr_1 and arr_2
    arr_1_copy = arr_1.copy()
    arr_2_copy = arr_2.copy()

    #Copy roi_mask if it is provided, threshold it so that pixels with positive values are set to 1 and background pixels are set to 0
    if hasattr(roi_mask, "__len__"):
        roi_mask_copy = np.where(roi_mask>0, 1, 0)

    #Initialize a dictionary to collect quantification results - it will be used to create the output pandas dataframe
    quantifications_dict = {'timepoint':[], "spearman_coeff_observed":[], "spearman_p_val_observed":[], "spearman_coeff_shuffle_mean":[],
                            "spearman_coeff_shuffle_std":[], "spearman_p_val_shuffle_mean":[], "spearman_p_val_shuffle_std":[]}

    #Add a variable in quantifications_dict for the spearman coefficient and a column for the p-value per each shuffle iteration time
    for s in range(shuffle_times):

        #Create column names - note: these names will be the columns of the output pandas dataframe
        column_name_shuffle_coeff = 'spearman_coeff_shuffle_'+str(s)
        column_name_shuffle_pval = 'spearman_p_val_shuffle_'+str(s)

        #Add columns name to the quantifications_dict and link them to an empty list
        quantifications_dict[column_name_shuffle_coeff]=[]
        quantifications_dict[column_name_shuffle_pval]=[]

    #Transform the 3D input arrays in lists of 2D arrays, depending on the axis to analyse. Transform roi_mask if provided
    if axis_to_use==0:
        arr1_list_2D = [arr_1_copy[tp,...] for tp in range(arr_1_copy.shape[0])]
        arr2_list_2D = [arr_2_copy[tp2,...] for tp2 in range(arr_2_copy.shape[0])]
        if hasattr(roi_mask, "__len__"):
            roi_mask_list_2D = [roi_mask_copy[tp3,...] for tp3 in range(roi_mask_copy.shape[0])]

    elif axis_to_use==1:
        arr1_list_2D = [arr_1_copy[:,tp4,:] for tp4 in range(arr_1_copy.shape[1])]
        arr2_list_2D = [arr_2_copy[:,tp5,:] for tp5 in range(arr_2_copy.shape[1])]
        if hasattr(roi_mask, "__len__"):
            roi_mask_list_2D = [roi_mask_copy[:,tp6,:] for tp6 in range(roi_mask_copy.shape[1])]

    elif axis_to_use==2:
        arr1_list_2D = [arr_1_copy[...,tp7] for tp7 in range(arr_1_copy.shape[2])]
        arr2_list_2D = [arr_2_copy[...,tp8] for tp8 in range(arr_2_copy.shape[2])]
        if hasattr(roi_mask, "__len__"):
            roi_mask_list_2D = [roi_mask_copy[...,tp9] for tp9 in range(roi_mask_copy.shape[2])]

    #Iterate through 2D arrays
    for progr, arr_1_image in enumerate(arr1_list_2D):
        quantifications_dict['timepoint'].append(progr)

        #If roi_mask if provided, only quantify the correlation of the 2D arrays if the number of pixels in the roi is higher than pixels_threshold
        #Initialize a variable to determine if the 2D array should be quantified. Set it to True (the array should be quantified)
        quantify_2D_arrays = True
        if hasattr(roi_mask, "__len__"):
            #Get the number of pixels in the roi_mask
            roi_mask_pixels_n = np.sum(roi_mask_copy)
            #Set quantify_2D_arrays to False if the number of pixels in roi_mask does not pass the threshold
            if roi_mask_pixels_n<=pixels_threshold:
                quantify_2D_arrays = False

        #Quantify the correlation between the 2D arrays if quantify_2D_array is still set to True
        if quantify_2D_arrays:
            #Get arr_2 image
            arr_2_image = arr2_list_2D[progr]

            #Smooth images if required
            if smooth_image:
                smooth_arr_1_image = blur_image(arr_1_image, n=gaus_n)
                smooth_arr_2_image = blur_image(arr_2_image, n=gaus_n)
            else:
                smooth_arr_1_image = arr_1_image
                smooth_arr_2_image = arr_2_image

            #Get roi_mask image and use it when calculating the correlation coefficient and pvalues, if it is provided
            if hasattr(roi_mask, "__len__"):
                roi_mask_image = roi_mask_list_2D[progr]

                #Get correlation coefficient and pvaue
                corr_arr1_arr2 = analyse_spearman_array(smooth_arr_1_image,
                                                        smooth_arr_2_image,
                                                        roi_mask=roi_mask_image,
                                                        smooth_image_1=False,
                                                        gaus_n_1=3,
                                                        gaus_ny_1=None)
            else:
                #Get correlation coefficient and pvaue
                corr_arr1_arr2 = analyse_spearman_array(smooth_arr_1_image,
                                                        smooth_arr_2_image,
                                                        roi_mask=roi_mask,
                                                        smooth_image_1=False,
                                                        gaus_n_1=3,
                                                        gaus_ny_1=None)

            #Add correlation coefficient and pvaue to the collection dictionary
            quantifications_dict['spearman_coeff_observed'].append(corr_arr1_arr2[0])
            quantifications_dict['spearman_p_val_observed'].append(corr_arr1_arr2[1])

            #Initialize lists to collect shuffle results
            shuffle_results_coeff_list = []
            shuffle_results_pval_list = []

            #Shuffle the results as many times as shuffle_times
            for i in range(shuffle_times):
                
                #Only consider roi pixels if roi_mask is provided
                if hasattr(roi_mask, "__len__"):
                    #Shuffle the arrays of the dirrefent channels. The output of the shuffling is a list
                    shff_list1 = sample(list(smooth_arr_1_image[roi_mask_image>0]), k=len(list(smooth_arr_1_image[roi_mask_image>0])))
                    shff_list2 = sample(list(smooth_arr_2_image[roi_mask_image>0]), k=len(list(smooth_arr_2_image[roi_mask_image>0])))
                else:
                    shff_list1 = sample(list(smooth_arr_1_image.flatten()), k=len(list(smooth_arr_1_image.flatten())))
                    shff_list2 = sample(list(smooth_arr_2_image.flatten()), k=len(list(smooth_arr_2_image.flatten())))

                #Transform shuffled lists into numpy arrays
                shff_arr1 = np.asarray(shff_list1)
                shff_arr2 = np.asarray(shff_list2)

                #Get correlation coefficient and pval
                shuff_corr_arr1_arr2 = spearmanr(shff_arr1, shff_arr2)
                
                #Append correlation coefficient and pval to their respective collection lists
                shuffle_results_coeff_list.append(shuff_corr_arr1_arr2[0])
                shuffle_results_pval_list.append(shuff_corr_arr1_arr2[1])

                #Add correlation coefficient and pval to the quantifications_dict
                column_name_i_shuffle_coeff = 'spearman_coeff_shuffle_'+str(i)
                column_name_i_shuffle_pval = 'spearman_p_val_shuffle_'+str(i)
                quantifications_dict[column_name_i_shuffle_coeff].append(shuff_corr_arr1_arr2[0])
                quantifications_dict[column_name_i_shuffle_pval].append(shuffle_results_pval_list)

            #Calculate mean and std
            shuffle_results_coeff_mean, shuffle_results_coeff_std = np.mean(shuffle_results_coeff_list), np.std(shuffle_results_pval_list)
            shuffle_results_pval_mean, shuffle_results_pval_std = np.mean(shuffle_results_pval_list), np.std(shuffle_results_pval_list)

            #Add quantification to quantifications_dict
            quantifications_dict['spearman_coeff_shuffle_mean'].append(shuffle_results_coeff_mean)
            quantifications_dict['spearman_coeff_shuffle_std'].append(shuffle_results_coeff_std)
            quantifications_dict['spearman_p_val_shuffle_mean'].append(shuffle_results_pval_mean)
            quantifications_dict['spearman_p_val_shuffle_std'].append(shuffle_results_pval_std)

        #Add NaN values to quantifications_dict if the 2D array is not quantified
        else:
            quantifications_dict['spearman_coeff_observed'].append(np.NaN)
            quantifications_dict['spearman_p_val_observed'].append(np.NaN)
            quantifications_dict['spearman_coeff_shuffle_mean'].append(np.NaN)
            quantifications_dict['spearman_coeff_shuffle_std'].append(np.NaN)
            quantifications_dict['spearman_p_val_shuffle_mean'].append(np.NaN)
            quantifications_dict['spearman_p_val_shuffle_std'].append(np.NaN)
            #Add NaN values per each shuffle time
            for s1 in range(shuffle_times):
                column_name_s1_shuffle_coeff = 'spearman_coeff_shuffle_'+str(s1)
                column_name_s1_shuffle_pval = 'spearman_p_val_shuffle_'+str(s1)
                quantifications_dict[column_name_s1_shuffle_coeff].append(np.NaN)
                quantifications_dict[column_name_s1_shuffle_pval].append(np.NaN)

    #Use quantifications_dict to form a pandas dataframe
    output_dataframe = pd.DataFrame.from_dict(quantifications_dict)

    if return_dict:
        return output_dataframe, quantifications_dict
    else:
        return output_dataframe


