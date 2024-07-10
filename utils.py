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

def get_distances(coords_1, coords_2, desired_distance='min'):
    """
    only tested on 2D arrays
    """
    #Get distances
    coords_distances = distance.cdist(coords_1, coords_2, 'euclidean')

    if desired_distance=='min':
        min_dist_axis_0 = np.argmin(coords_distances, axis=0)[0]
        min_dist_axis_1 = np.argmin(coords_distances, axis=1)[0]

    return
