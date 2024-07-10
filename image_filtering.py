import numpy as np
from scipy.signal import convolve
from scipy.spatial import KDTree
from skimage.filters import median as medianfilter
from skimage.filters import frangi
from skimage.util import img_as_float32
from skimage.measure import label, regionprops
import cv2


def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions - this code is taken from scipy documentation https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html"""
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()


def blur_image(im, n, ny=None):
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
        This code modified from scipy documentation https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """
    #Copy the input image
    im_copy = im.copy()
    g = gauss_kern(n, sizey=ny)
    improc = convolve(im_copy,g, mode='same')
    return(improc)

def median_blur_image(imm, **kwargs):
    """
    returns a smoothed copy of input image using a median filter
    """
    #Copy input image
    imm_copy = imm.copy()
    return medianfilter(imm_copy, **kwargs)


def bilateral_filter_image(img, smooth_diameter, smooth_sigma_color, smooth_sigma_space):
    """
    returns a smoothed copy of the input image using a bilateral filter
    """
    #Copy input image
    img_copy = img.copy()

    #Transform input image to type float32 before applying bilateral filtering
    img_f32 = img_as_float32(img_copy, )
    #Apply bilateral filtering
    img_bilat = cv2.bilateralFilter(img_f32, smooth_diameter, smooth_sigma_color, smooth_sigma_space)
    return img_bilat

def frangi_filter(immg, **kwargs):
    """
    returns a copy of the filtered input image after applying a frangi filtering
    """
    #Copy input image
    immg_copy = immg.copy()

    return frangi(immg_copy, **kwargs)


def highpass_area_filter(input__binary__imag_e, area_highpass_thr, return_area_list=False, input_is_label_image=False,
                         output_lowval=0, output_highval=255, output_dtype=np.uint8):
    """
    Given a binary input array and a threshold for area (the threshold is in number of pixels), it filters the structures of the binary image and only keep those whose area is
    higher than the threshold.
    
    By default input__binary__imag_e must be a binary mask. It is however possible to alternatively provide a label image.
    If a label image is provided for input__binary__imag_e, the parameter input_is_label_image must be set to True.

    NOTE: the option of providing input__binary__imag_e as a label image hasn't been properly tested.

    The default output is a binary mask of values 0, 255 and dtype uint8.
    """
    #Copy the input image
    input__binary__imag_e_copy = input__binary__imag_e.copy()

    # label image regions if input image is not already a label image
    if input_is_label_image:
        label__im_g = input__binary__imag_e.copy()
    else:
        label__im_g = label(input__binary__imag_e_copy)
    
    #measure the properties of the region
    label__im_g_properties = regionprops(label__im_g)

    #Initialize a zero array to be modified as output array
    out_put_arr_ay = np.zeros((input__binary__imag_e_copy.shape[0], input__binary__imag_e_copy.shape[1])).astype(np.uint8)

    #Initialize a collection list for the areas
    areas_cl = []
    
    #Iterate through the regions of the labelled image, identified using measure.regionprops
    for re_gi_on in label__im_g_properties:
        
        #Get the area of the region
        re_gion_area = re_gi_on.area

        #Add area to collection list
        areas_cl.append(re_gion_area)

        #If the region area is higher than the highpass area threshold, modify the output array
        if re_gion_area >= area_highpass_thr:
    
            #Get region coordinates
            re_gi_on_coordinates = re_gi_on.coords
    
            #Unzip the coordinates in individual lists
            unzipped_re_gi_on_coordinates = [list(t) for t in zip(*re_gi_on_coordinates)]
            
            #Set output array values at region coordinates to 255
            out_put_arr_ay[unzipped_re_gi_on_coordinates[0], unzipped_re_gi_on_coordinates[1]] = 255
    
    #Rescale output array in the desired range
    rescaled_out_put_arr_ay = np.where(out_put_arr_ay>0, output_highval, output_lowval).astype(output_dtype)

    #Return output array and area list if return_area_list is selected, else only return the output array
    if return_area_list:
        return rescaled_out_put_arr_ay, areas_cl
    else:
        return rescaled_out_put_arr_ay



def filter_mask1_on_mask2(mask_1, mask_2, pixels_highpass_threshold=0, output_lowval=0, output_highval=255, output_dtype=np.uint8):
    """
    Given a binary mask_1 and second binary mask_2, the function:
    1) iterates through the individual regions of mask_1 (individual regions are areas of pixels which are entirely surrounded by background).
    2) keeps the region if at least a number of pixels higher than pixels_highpass_threshold (default 1) overlaps with pixels in mask_2

    For both mask_1 and mask_2 positive pixels are assumed to be the pixels of interest

    The default output is a binary mask of values 0, 255 and dtype uint8.

    """
    #Copy mask_1 and mask_2
    mask_1_copy = mask_1.copy()
    mask_2_copy = mask_2.copy()

    #Label regions in mask_1
    label_mask_1 = label(mask_1_copy)
    
    #measure the properties of the regions in mask_1
    regionprops_mask_1 = regionprops(label_mask_1)

    #Get coordinates of mask_2 positive pixels - reorganize them to be a list of tuples
    coord_mask_2 = np.argwhere(mask_2_copy>0)
    coord_mask_2_as_list_of_tuples = [(cr_dnt[0], cr_dnt[1]) for cr_dnt in list(coord_mask_2)]

    #Initialize a zero array to be modified as output array
    output_array_filtered_img = np.zeros((mask_1_copy.shape[0], mask_1_copy.shape[1])).astype(np.uint8)

    #Iterate through the regions of mask_1, identified using regionprops
    for m1_reg_i_on in regionprops_mask_1:
    
        #Get region coordinates - reorganize them to be a list of tuples
        m1_reg_i_on_coordinates = m1_reg_i_on.coords
        m1_reg_i_on_coordinates_as_listoftupl = [(cr_dnt1[0], cr_dnt1[1]) for cr_dnt1 in list(m1_reg_i_on_coordinates)]
    
        #Get intersection of region coordinates and mask_2 positive-pixels coordinates
        m1_reg_i_on_interescion_with_m2 = list(set(m1_reg_i_on_coordinates_as_listoftupl).intersection(set(coord_mask_2_as_list_of_tuples)))

        #If there is an intersection, add the region to the output array
        if len(m1_reg_i_on_interescion_with_m2)>pixels_highpass_threshold:
            #Unzip the coordinates in individual lists
            unzipped_m1_reg_i_on_coordinates = [list(tt33) for tt33 in zip(*m1_reg_i_on_coordinates)]
        
            #Set output array values at region coordinates to 255
            output_array_filtered_img[unzipped_m1_reg_i_on_coordinates[0], unzipped_m1_reg_i_on_coordinates[1]] = 255

    # #Rescale output array in the desired range
    rescaled_output_array_filtered_img = np.where(output_array_filtered_img>0, output_highval, output_lowval).astype(output_dtype)
    
    return rescaled_output_array_filtered_img


def filter_mask1_by_centroid_distance_from_mask2(mask_1_img, mask_2_img, distance_thr, filtering_modality='highpass', n_distances=2, return_coordinates=False,
                                                 output_low_value=0, output_high_value=255, output_dtype=np.uint8):
    """
    Given a binary mask_1 and a second binary mask_2, the function:
    1) iterates through individual regions of the mask_1 (individual regions are areas of pixels which are entirely surrounded by background).
    2) Per each region calculates the coordinates of the centroid.
    3) Keeps the region only if...
        3A) if the filtering_modality is 'highpass', regions whose centroid distance from the closest pixels in mask_2 is higher than distance_thr are kept.
        3B) if the filtering_modality is 'lowpass', regions whose centroid distance from the closest pixels in mask_2 is lower than distance_thr are kept.

    For both mask_1 and mask_2 positive pixels are assumed to be the pixels of interest.

    The default output is a binary mask of values 0, 255 and dtype uint8.   

    """
    #Copy mask_1_img and mask_2_img
    mask_1_img_copy = mask_1_img.copy()
    mask_2_img_copy = mask_2_img.copy()

    #label mask_1 regions
    label_mask_1_img = label(mask_1_img_copy)
    
    #measure the properties of the region
    mask_1_img_regionprops = regionprops(label_mask_1_img)

    #Get the coordinates of mask_2 regions
    mask_2_img_coord_rowY_colX = np.argwhere(mask_2_img_copy>0) #THIS IS CORRECTLY NAMED ROW_Y, COL_X
        
    #Transform coordinates so that they are tuples in a list
    mask_2_img_coord_colX_rowY_listoftuples_i = [(list(cccrd)[1], list(cccrd)[0]) for cccrd in mask_2_img_coord_rowY_colX]  #WHEN PASSED TO A PLOT WHICH EXPECTS COL_X, ROW_Y, THE RESULT IS CORRECT!!! SO THIS IS CORRECTLY NAMED!

    #Initialize a zero array to be modified as output array
    output_arr_ay = np.zeros((mask_1_img_copy.shape[0], mask_1_img_copy.shape[1])).astype(np.uint8)

    #For visualization purposes it could be convenient to return the centroid linked to the closest point in mask_2. If return_coordinates is specified,
    #Initialize a dictionary to link the centroid coordinates of regions in mask_1 to those of its closest point in mask_2
    if return_coordinates:
        coordinates_linking_dict = {}
    
    # #Initialize a centroids coord collection list
    # centroid_coord_coll_list = []

    #Iterate through the regions of the labelled image, identified using measure.regionprops
    for re_gion in mask_1_img_regionprops:
        
        #Re-initialize the coordinates of mask2
        mask_2_img_coord_colX_rowY_listoftuples = mask_2_img_coord_colX_rowY_listoftuples_i

        #Get the coordinates of the centroid of the region
        cc_yy_row, cc_xx_col = re_gion.centroid #WHEN PASSED TO A PLOT WHICH EXPECTS COL_X, ROW_Y, THE RESULT IS INVERTED!!! SO THIS IS CORRECTLY NAMED

        #Invert centroid region coordinates to match the list of coordinate for mask_2
        input_re_gion_centroid = (cc_xx_col, cc_yy_row) #WHEN PASSED TO A PLOT WHICH EXPECTS COL_X, ROW_Y, THE RESULT IS CORRECT!!! SO THIS IS CORRECTLY NAMED

        #Add centroid coordinates to the list of coordinates of mask_2
        mask_2_img_coord_colX_rowY_listoftuples_plusCC = mask_2_img_coord_colX_rowY_listoftuples + [input_re_gion_centroid]
        
        # Make a tree out of the outside embryo coordinates plus the centroid coordinates
        mask_2_coord_tree = KDTree(mask_2_img_coord_colX_rowY_listoftuples_plusCC)
        
        #Get the n closest distances (and relative indeces in the list of coordinates) to the region centroid. NOTE: the first (aka closest) distance is always the point itself with distance 0
        distance__m1_m2, result__m1_m2 = mask_2_coord_tree.query(input_re_gion_centroid, k=n_distances)
        
        #If filtering_modality is highpass, the region is kept when its centroid is further apart than distance_thr
        if filtering_modality=='highpass':
            
            #If the nearest neightbor distance of the region is higher than the highpass threshold, modify the output array
            if list(distance__m1_m2)[1]>distance_thr:
                
                #Get region coordinates
                re_gion_coordinates = re_gion.coords
        
                #Unzip the coordinates in individual lists
                unzipped_re_gion_coordinates = [list(t) for t in zip(*re_gion_coordinates)]
                
                #Set output array values at region coordinates to 255
                output_arr_ay[unzipped_re_gion_coordinates[0], unzipped_re_gion_coordinates[1]] = 255
        

        #If filtering_modality is lowpass, the region is kept when its centroid is closer than distance_thr
        elif filtering_modality=='lowpass':
            #If the nearest neightbor distance of the region is lower than the lowpass threshold, modify the output array
            if list(distance__m1_m2)[1]<distance_thr:
                
                #Get region coordinates
                re_gion_coordinates = re_gion.coords
        
                #Unzip the coordinates in individual lists
                unzipped_re_gion_coordinates = [list(t) for t in zip(*re_gion_coordinates)]
                
                #Set output array values at region coordinates to 255
                output_arr_ay[unzipped_re_gion_coordinates[0], unzipped_re_gion_coordinates[1]] = 255

        #For visualization purposes it could be convenient to return the centroid linked to the closest point in mask_2
        if return_coordinates:
            #Get the coordinates of closest point to the centroid in mask_2
            cpm2__coord = mask_2_img_coord_colX_rowY_listoftuples_plusCC[list(result__m1_m2)[1]]
            #Link the centroid coordinates and the closest point coordinates in the output dictionary
            coordinates_linking_dict[input_re_gion_centroid]=cpm2__coord #THESE SHOULD BE MATCHING AND BOTH BE IN THE FORMAT COL_X, ROW_Y
    
    #Rescale output array in the desired output range
    rescaled_output_arr_ay = np.where(output_arr_ay>0, output_high_value, output_low_value).astype(output_dtype)

    #For visualization purposes it could be convenient to return the centroid linked to the closest point in mask_2
    if return_coordinates:
        return rescaled_output_arr_ay, coordinates_linking_dict
        
    else:
        return rescaled_output_arr_ay


