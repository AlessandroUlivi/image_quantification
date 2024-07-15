import os
import numpy as np
from skimage.draw import polygon
from roifile import ImagejRoi
from scipy.spatial import distance


def listdirNHF(input_directory):
    """
    creates a list of files in the input directory, avoiding hidden files. Hidden files are identified by the fact that they start with a .
    input: directory of the folder whose elements have to be listed.
    output: list of elements in the input-folder, except hidden files. 
    """
    return [f for f in os.listdir(input_directory) if not f.startswith(".")]


def form_mask_from_roi(roi_file_path, reference_img, return_coordinates=False, roi_pixel_value=255, background_pixel_val=0, output_dtype=np.uint8):
    """
    Given:
    - The directory of a .roi or .zip file containing one or multiple ROIs saved from Fiji/ImageJ.
    - A reference image to be used for shape.

    The function returns a binary array of the same shape as reference, where pixels of the ROIs are assinged value roi_pixel_value (default 255) and the rest of the pixels are
    assigned value background_pixel_val (default 0). The default output dtype is uint8.

    If return_coordinates=True (default is False) the function returns the coordinates of the .roi/.zip file. If the file is a signle roi (extension .roi) the output is a numpy array
    of sub-numpy-arrays. Each sub-numpy-array is the col_x, row_y coordinates of a pixel of the roi. If the roi file contains multiple rois (extension .zip) the output is a list where
    each element is a numpy array of sub-numpy-arrays. Each numpy-array corresponds to an individual roi. Each sub-numpy-array of each numpy array is the col_x, row_y coordinates of a
    pixel of the individual roi.

    NOTES: the function is tested for Imagej/Fiji -generated roi files (extensions .roi or .zip). The function was only tested on 2D images.
    """

    #Open roi file
    roi_file = ImagejRoi.fromfile(roi_file_path)

    #Initialize the output array as a zero array of the same shape of reference_img
    out_img = np.zeros((reference_img.shape[0], reference_img.shape[1])).astype(np.uint8)

    #Iterate through the coordinates of the roi file and collect col_x and row_y pixels coordinates in separate lists.
    # Note: if file is a signle roi (extension .roi) roi_file_copy is a list of tuples. Each tuple is the col_x, row_y coordinates of a pixel of the roi.
    # If the roi file contains multiple rois (extension .zip) the file is a list where each element is a sub-list of tuples. Each sub-list corresponds to an individual roi.
    # Each tuple of each sub-list is the col_x, row_y coordinates of a pixel of the individual roi.

    #If the input file is a collection of rois
    if roi_file_path[-4:]==".zip":

        #Initialize an outputlist to collect coordinates of different rois, if return_coordinates is set to True
        if return_coordinates:
            output_coords_coll_list = []

        #Iterate through the individual sub-roi
        for sub_roi in roi_file:

            #Initialize collection lists for col_x and row_y coordinates
            col_x_list = []
            row_y_list = []

            #Get roi_file_coordinates
            sub_roi_file_coords = sub_roi.coordinates()

            #Add coordinates to output collection list if return_coordinates is set to True
            if return_coordinates:
                output_coords_coll_list.append(sub_roi_file_coords)

            #Iterate through the coordinates of the individual sub-roi
            for c in sub_roi_file_coords:
                col_x_list.append(c[0]-1) #Note: because roi have been generate in ImageJ their coordinate numeration starts from 0, while python starts from 0
                row_y_list.append(c[1]-1) #Note: because roi have been generate in ImageJ their coordinate numeration starts from 0, while python starts from 0

            #Transform the coordinates collection lists in numpy arrays
            col_x_array = np.asarray(col_x_list)
            row_y_array = np.asarray(row_y_list)

            #Use the array coordinates to form a polygon
            yy_rr, xx_cc = polygon(row_y_array, col_x_array)

            #Modify the output array
            out_img[yy_rr, xx_cc]=255

    #If the input file is a single rois
    else:
        #Initialize collection lists for col_x and row_y coordinates
        col_x_list_1 = []
        row_y_list_1 = []
        
        #Get roi_file_coordinates
        roi_file_coords = roi_file.coordinates()
        
        #Iterate through the coordinates of the roi
        for c1 in roi_file_coords:
            col_x_list_1.append(c1[0]-1) #Note: because roi have been generate in ImageJ their coordinate numeration starts from 0, while python starts from 0
            row_y_list_1.append(c1[1]-1) #Note: because roi have been generate in ImageJ their coordinate numeration starts from 0, while python starts from 0
    
        #Transform the coordinates collection lists in numpy arrays
        col_x_array_1 = np.asarray(col_x_list_1)
        row_y_array_1 = np.asarray(row_y_list_1)

        #Use the array coordinates to form a polygon
        yy_rr_1, xx_cc_1 = polygon(row_y_array_1, col_x_array_1)

        #Modify the output array
        out_img[yy_rr_1, xx_cc_1]=255

    #Rescale the output array in the desired range
    rescaled_out_img = np.where(out_img>0, roi_pixel_value, background_pixel_val).astype(output_dtype)

    if return_coordinates:
        if roi_file_path[-4:]==".zip":
            return rescaled_out_img, output_coords_coll_list
        else:
            return rescaled_out_img, roi_file_coords
    else:
        return rescaled_out_img

def get_euclidean_distances(coords_1, coords_2, desired_distance='min'):
    """
    Given the coordinates of the pixels of non-overlapping regions (lists of tuples, each tuple corresponding to one pixel), the function returns:
    - if desired_distance = 'min', a tuple with in position 0 the maximum euclidean distance between the two regions (a.u., float) and in position 1
    the coordinates of the two pixels for which the distance is calculated (list of tuples, position 0 are the coordinates of pixel in coords_1, position 1 the coordinates of pixel in 
    coords_2).
    - if desired_distance = 'max', a tuple with in position 0 the maximum euclidean distance between the two regions (a.u., float) and in position 1
    the coordinates of the two pixels for which the distance is calculated (list of tuples, position 0 are the coordinates of pixel in coords_1, position 1 the coordinates of pixel in 
    coords_2).
    - if desired_distance = 'mean', a tuple with in position 0 the mean euclidean distance between the pixels of the two regions (a.g., float) and None in position 1.

    desired_distance is set to 'min' as default.

    If multiple pixels have the same min or max distance, only one is returned. It is the first value found when iterating along the coordinates list.
    """
    #Double check that desired_distance is either 'min', or 'max', or 'mean'
    assert desired_distance in ['min', 'max', 'mean'], "desired_distance must be either 'min', or 'max', or 'mean'"

    #Get distance matrix between input coordinates
    coords_distances = distance.cdist(coords_1, coords_2, 'euclidean')

    #Get the minimum distance and relative pixels if desired_distance is set to 'min'
    if desired_distance=='min':
        #Get indexes of minimum distance in the coords_distances matrix for each axis
        min_dist_axis_0 = np.argmin(coords_distances, axis=0)
        min_dist_axis_1 = np.argmin(coords_distances, axis=1)

        #Initialize the output variables: the min_distance between coords_1 and coords_2, the coordinates of the pixel in coords_1 closest to pixels in coords_2 and the
        #the coordinates of the pixel in coords_2 closest to pixels in coords_1
        min_distance = np.max(coords_distances) #to guarantee that the next loop will lead to the identification of a the min value, despite potential missmatches between np.min calculation and distance.cdist calculation, initialize the variable as the max in the distance matrix
        closest_px_1 = tuple(0 for c1 in range(len(coords_1[0]))) #initialize as a series of 0 coordinates per each dimension of coords_1
        closest_px_2 = tuple(0 for c2 in range(len(coords_2[0]))) #initialize as a series of 0 coordinates per each dimension of coords_2

        #Iterate through the indexes of the minima found along each axis, get the corresponding value and update min_distance if the found value is smaller than the current
        #Also update the pixels coordinates when a min value is found
        for i in min_dist_axis_0:
            for j in min_dist_axis_1:
                ij_val = coords_distances[i][j]
                if ij_val < min_distance:
                    min_distance = ij_val
                    closest_px_1 = coords_1[i]
                    closest_px_2 = coords_2[j]

        return min_distance, [closest_px_1, closest_px_2]

    #Get the minimum distance and relative pixels if desired_distance is set to 'max'
    elif desired_distance=='max':
        #Get indexes of maximum distance in the coords_distances matrix
        max_dist_axis_0 = np.argmax(coords_distances, axis=0)
        max_dist_axis_1 = np.argmax(coords_distances, axis=1)

        #Initialize the output variables: the max_distance between coords_1 and coords_2, the coordinates of the pixel in coords_1 furthest to pixels in coords_2 and the
        #the coordinates of the pixel in coords_2 furthest to pixels in coords_1
        max_distance = np.min(coords_distances) #to guarantee that the next loop will lead to the identification of a the max value, despite potential missmatches between np.max calculation and distance.cdist calculation, initialize the variable as the min in the distance matrix
        furthest_px_1 = tuple(0 for c1 in range(len(coords_1[0]))) #initialize as a series of 0 coordinates per each dimension of coords_1
        furthest_px_2 = tuple(0 for c2 in range(len(coords_2[0]))) #initialize as a series of 0 coordinates per each dimension of coords_2

        #Iterate through the indexes of the minima found along each axis, get the corresponding value and update min_distance if the found value is smaller than the current
        #Also update the pixels coordinates when a min value is found
        for k in max_dist_axis_0:
            for w in max_dist_axis_1:
                kw_val = coords_distances[k][w]
                if kw_val > max_distance:
                    max_distance = kw_val
                    furthest_px_1 = coords_1[k]
                    furthest_px_2 = coords_2[w]

        return max_distance, [furthest_px_1, furthest_px_2]

    #Get the average distance if desired_distance is set to 'mean'
    else:
        #Get the average distance in the coords_distances matrix
        mean_distance = np.mean(coords_distances)

        return mean_distance, None

def combine_rois(input_rois, thresholds=0, i_axis=None, binarize_output=True, output_lowval=0, output_highval=255, output_dtype=np.uint8):
    """
    Given:
    - input_rois, list of ndarrays or ndarray. If list of ndarrays each ndarray must have the same shape. If single ndarray, i_axis must be provided.
    - thresholds, float, int or list of floats/int. If list, the len of the list must match the len of input_rois (if input_rois is a list) or the size of ndarray's i_axis
    (if input_rois is an ndarray).

    The function:
    1) if input_rois is an ndarray, it is splat in m ndarrays along the i_axis, where m correspond to the size of the i_axis. If input_rois is already a list, nothing is done.
    Output list-1.
    2) iterates through the individual ndarrays (arr[i]) of list-1.
    3) Binarizes each arr[i] using a highpass threshold (thre[i]). If thresholds is an int or a float, thre[i]=thresholds and it is applied, identical, to all arr[i].
    If thresholds is a list. thre[i] corresponds to the float/int in thresholds which has the same index of arr[i]. When binarizing, it is assumed that pixels of arr[i] whose value is
    >thre[i] are the pixels of interest (set to 1) and the rest the background (set to 0).
    4) Joins all arr[i] in a single output array (out_arr) of the same shape of arr[i], by summing their values.
    5) If binarize_output==True (default), out_arr is binarized by setting all positive pixels to output_highval and all non positive pixels to output_low_val. If binarize_output==False,
    out_arr is returned.
    """

    #Use input_rois as it is if a list is provided
    if isinstance(input_rois, list):
        iteration_list = input_rois
    
    #If input_rois is a ndarray
    elif hasattr(input_rois, "__len__"):

        #Make sure that i_axis is provided
        assert i_axis!=None, "provide a value for i_axis if input_roi is an array"

        #Split input_rois in m ndarray along i_axis, where m equals to the size of i_axis in input_rois
        iteration_list = [np.squeeze(a) for a in np.split(input_rois, indices_or_sections=input_rois.shape[i_axis], axis=i_axis)]
        
    
    #Initialize an output array
    output_array = np.zeros(iteration_list[0].shape)
    
    #Iterate along the iteration list
    for pos, arr in enumerate(iteration_list):
        
        #If thresholds is a list, get the thresholding value from the list, taking the value at the same index of the array under iteration
        if isinstance(thresholds, list):
            t = thresholds[pos]
        #If thresholds is a single number, use it as a threshold
        else:
            t = thresholds

        #Binarize the array under iteration by setting pixels whose value is >threshold to 1 and the rest to 0
        bin_arr = np.where(arr>t, 1, 0)

        #Update the output array
        output_array += bin_arr

    #Binarize the output_array and set the values in the wanted range, if binarize_output is True. Return it as is otherwise
    if binarize_output:

        final_output_array = np.where(output_array>0, output_highval, output_lowval).astype(output_dtype)
        return final_output_array
    else:
        return output_array


def mantain_exclude_image_rois():
    """
    given an roi to maintain and an roi to exclude, it combines them in the segmentation of an image.
    """
    return

