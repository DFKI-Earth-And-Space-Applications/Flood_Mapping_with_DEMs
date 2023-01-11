import os
import json
import numpy as np
import shutil
from glob import glob
import zipfile
import warnings
import argparse
import gdal
from glob import glob
from scipy import interpolate
import math
import elevation


from typing import List
import sys

# create a .wgetrc file with "touch .wgetrc | chmod og-rw .wgetrc" and enter your credentials. Guide is here: https://lpdaac.usgs.gov/resources/e-learning/how-access-lp-daac-data-command-line/ (2. Download LP DAAC data from the command line with wget)
# according to the readme https://github.com/cloudtostreet/Sen1Floods11 catalog information is STAC compliant
# definitions of STAC specs are here: https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md
# sen1floods11 dataset: All images are projected to WGS 84 (EPSG:4326) at 10 m ground resolution
# STAC uses World Geodetic System 1984 (WGS 84) https://datatracker.ietf.org/doc/html/rfc7946
# STAC and json files of sen1floods11 use longitude as first value and latitude as second one https://datatracker.ietf.org/doc/html/rfc7946#appendix-A.1
# ASTER referenced to the WGS84/EGM96 geoid
# SRTM relative to the WGS84 ellipsoid


def get_DEM(in_path, out_path):
    # this for ASTER
    print('Get relevant file paths from input folder')
    file_paths = get_file_paths(in_path)
    print('Extract coordinate files from input data tiles')
    coordinates = generate_file_name_from_coordinates(file_paths)
    print('Download DEM data')
    generate_DEM_data(out_path, file_paths, coordinates, aster=True, srtm=False)
    print('Unzip downloaded DEM data')
    unzip_generated_DEM_data(out_path)


def get_DEM_SRTM(in_path, out_path):
    elevation.DEFAULT_PRODUCT = elevation.PRODUCTS[1]
    print('Get relevant file paths from input folder')
    file_paths = get_file_paths(in_path)
    print('Extract coordinate files from input data tiles')
    coordinates = get_corners(file_paths)
    print('Download DEM data')
    download_SRTM(out_path, file_paths, coordinates)


def unzip_generated_DEM_data(tar_path):
    # unzip all downloaded zip files in a specified folder by iterating through all subfolders
    dirs = glob(os.path.join(tar_path, "*", ""))
    for dir in dirs:
        files = os.listdir(dir)
        for file in files:
            if file.endswith(".zip"):
                path_to_zip_file = os.path.join(dir, file)
                with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                    zip_ref.extractall(dir)
        if len(files) == 1:
            warnings.warn(
                "Warning, no files downloaded for folder {}. Either, location does not exist in dataset or location "
                "is on the sea".format(
                    dir))


def download_SRTM(tar_path, tar_folders, coord_list):
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)
    else:
        sys.exit(f'Download folder {tar_path} already exists')
    for folder_path, coords in zip(tar_folders, coord_list):
        os_slash = os.path.join('a', 'b')[1]
        target = folder_path.split(os_slash)[-2]
        final_target = os.path.join(tar_path, target)
        os.makedirs(final_target)
        final_target_file = os.path.join(os.getcwd(), final_target, 'DEM.tif')
        # coords is of form: [left_bottom, right_top] where each elem defines lat and lon
        elevation.clip(bounds=(coords[0][0], coords[0][1], coords[1][0], coords[1][1]), output=final_target_file)
        elevation.clean()


def generate_DEM_data(tar_path, tar_folders, coord_list, aster=False, srtm=False, srtm_path=None):
    # download DEM data according to given coordinate lis into a given folder
    if not os.path.exists(tar_path):
        os.makedirs(tar_path)
    else:
        sys.exit(f'Download folder {tar_path} already exists')
    for folder_path, coords in zip(tar_folders, coord_list):
        os_slash = os.path.join('a', 'b')[1]
        target = folder_path.split(os_slash)[-2]
        final_target = os.path.join(tar_path, target)
        os.makedirs(final_target)
        for coord in coords:
            if aster:
                command = 'wget {download_link} -P {target_path}'.format(
                    download_link='https://e4ftl01.cr.usgs.gov/ASTT'
                                  '/ASTGTM.003/2000.03.01/ASTGTMV003_'
                                  + coord + '.zip',
                    target_path=final_target)
                os.system(command)
                command = 'wget {download_link} -P {target_path}'.format(
                    download_link='https://e4ftl01.cr.usgs.gov/ASTT'
                                  '/ASTGTM.003/2000.03.01/ASTGTMV003_'
                                  + coord + '.zip.xml',
                    target_path=final_target)
                os.system(command)
            if srtm:
                src_path = os.path.join(srtm_path, 'NASADEM_HGT_' + coord.lower() + '.zip')
                command = 'cp {src} {trg}'.format(src=src_path, trg=final_target)
                os.system(command)


def get_corners(file_list):
    # result will be a list of tuples
    # generate a tuple for each folder, containing the left bottom and right top coordinates, defining the files that
    # need to be downloaded from SRTM
    result = list()
    for file in file_list:
        # coordinates of a polygon might be in different tiles, therefore create a list for each polygon as well
        with open(file) as f:
            data = json.load(f)
            coords = data['geometry']['coordinates'][0]
            res_tuple = get_corner(coords)
            result.append(res_tuple)
    return result


def get_corner(coord):
    # Extract the left bottom and right top coordinates from 4 coordinates.
    # Input is in Longitude/Latitude
    # required is longitude and latitude as output
    # left bottom is smallest latitude and longitude
    # right top is biggest latitude and longitude
    # NOTE Sometimes it has 5 coordinates but one is a duplicate...
    coord = [(t[0], t[1]) for t in coord]
    tmp = list(set(coord))
    left_bottom = min(tmp)
    epsilon = 8.983152841195857e-03
    left_bottom = (left_bottom[0]-epsilon, left_bottom[1]-epsilon)
    right_top = max(tmp)
    right_top = (right_top[0]+epsilon, right_top[1]+epsilon)
    return [left_bottom, right_top]


def generate_file_name_from_coordinates(file_list):
    # generate a list for each folder, containing a list of all tiles that need to be downloaded from a datasource
    result = list()
    for file in file_list:
        # coordinates of a polygon might be in different tiles, therefore create a list for each polygon as well
        polygons = list()
        with open(file) as f:
            data = json.load(f)
            coords = data['geometry']['coordinates'][0]
            for coord in coords:
                file_name = transform_coordinates_into_file_name(coord)
                polygons.append(file_name)
            polygons = list(set(polygons))
            result.append(polygons)
    return result


def transform_coordinates_into_file_name(coord):
    # given a coordinate tuple of latitute and longitude outputs the filename of the tile containing these coordinates
    lat = coord[1]
    lon = coord[0]
    if lat >= 0:
        ns = 'N'
    elif lat < 0:
        lat = lat - 1
        ns = 'S'

    if lon >= 0:
        ew = 'E'
    elif lon < 0:
        lon = lon - 1
        ew = 'W'

    file_name = "%(ns)s%(lat)02d%(ew)s%(lon)03d" % \
                {'lat': abs(lat), 'lon': abs(lon), 'ns': ns, 'ew': ew}

    return file_name


def check_for_similarity(source_files, label_files):
    results_diff = list()
    for src_file, label_file in zip(source_files, label_files):
        with open(src_file) as src:
            with open(label_file) as label:
                src_data = json.load(src)
                label_data = json.load(label)
                src_coord = src_data['geometry']['coordinates'][0]
                label_coord = label_data['geometry']['coordinates'][0]
                diff = calc_diff(src_coord, label_coord)
                results_diff.append(diff)
    for vals in results_diff:
        for val in vals:
            for v in val:
                if v != 0:
                    if v > 0.0000000000001:  # checks for value greater than e^-14
                        print(v)
    return results_diff


def calc_diff(coords1, coords2):
    return np.array(coords1) - np.array(coords2)


def get_file_paths(target_path: str) -> List[str]:
    """
    :param target_path: defining the path where the location folders are contained
    :return: a sorted list of strings defining the path to the json file of each location
    """
    collection_file = os.path.join(target_path, 'collection.json')
    file_list = list()
    with open(collection_file) as f:
        data = json.load(f)
        # print(len(data['links'])) for whatever reason in label json file entries are contained multiple times...
        for elem in data['links']:
            rel = elem['rel']
            href = elem['href']
            if rel == 'item':
                # split based on directory slash given in json, so that we can rebuild the path independently of OS
                href_splitted = href.split('/')
                res_path = os.path.join(target_path, href_splitted[1], href_splitted[2])
                file_list.append(res_path)
    file_list = list(set(file_list))  # remove duplicate entries...
    file_list.sort()
    return file_list


def get_dem_tif_aster(tar_path, hand_path, res_out_path):
    print('Align DEM data with sen1floods11')
    # generate a tif file corresponding to the label lat/lon values with the DEM values
    dirs = glob(os.path.join(tar_path, "*", ""))
    # iterate through all sen1floods11 location folders
    for dir in dirs:
        print(f'Working on directory {dir}')
        files = os.listdir(dir)
        dem_files = list()
        # iterate through all files inside a location
        for file in files:
            # This is the convention behind an ASTER DEM file
            if file.endswith("_dem.tif") and file.startswith('ASTGTMV003_'):
                file_path = os.path.join(dir, file)
                dem_files.append(file_path)
                os_slash = get_os_slash()
                # need this if because folder names in labeled_source and labeled_label have different names and only
                # one name matches with the actual label file. Therefore change the name to the correct naming
                location_folder = file_path.split(os_slash)[-2].split('_')
                if location_folder[-1] == 'label':
                    location_folder = '_'.join(location_folder[:-1])
                else:
                    location_folder = '_'.join(location_folder)
                # generate path to corresponding label file
                label_folder_name = hand_path.split(get_os_slash())[-1]
                label_path = os.path.join(hand_path, location_folder + '_' + label_folder_name + '.tif')
                # generate output file name
                file_name = file_path.split(os_slash)[-2]
                out_path_short = os.path.join(res_out_path, file_name)
                out_path = os.path.join(out_path_short, file_name + '_dem.tif')
                out_path_merged = os.path.join(out_path_short, file_name + 'total_merged_dem.tif')
                if not os.path.exists(out_path_short):
                    os.makedirs(out_path_short)
                # label_path = os.path.join(hand_path, 'Bolivia_103757_LabelHand.tif')

        # read dem and label data
        dem_files.sort()
        # TODO: GENERAL sorting should help to ensure that some rare cases in appending DEM maps doesn't occur. Will
        # fail for general cases. E.g. imagine a row of three tiles. A fourth one might be to the top left. Sorting
        # might give this one as the first and the second one as the bottom rightmost one. They don't share a border
        # and therefore couldn't be merged in the next steps. Idea: use one start tile and order based on lowest
        # Levenshtein distance to start tile
        dem_datas = [gdal.Open(dem_file, gdal.GA_ReadOnly) for dem_file in dem_files]
        label_data = gdal.Open(label_path, gdal.GA_ReadOnly)
        # print(dem_data.GetProjection()) #projection method
        # print(dem_datas[0].GetGeoTransform()) #projection method
        # print(dem_data.RasterXSize) #X size
        # print(dem_data.RasterYSize) #Y size
        # print(dem_data.RasterCount) #number of band
        # print(dem_data.GetMetadata())#meta data
        dem_Xs, dem_Ys, dem_vals = list(), list(), list()
        # dem_Xst, dem_Yst, dem_valst = np.array(), np.array(), np.array()
        dem_X, dem_Y, dem_val = None, None, None
        for dem_data in dem_datas:
            # get the longitude and latitude values and the corresponding dem values of the DEM file
            dem_width = dem_data.RasterXSize
            dem_height = dem_data.RasterYSize
            dem_gt = dem_data.GetGeoTransform()
            dem_minx = dem_gt[0]
            dem_miny = dem_gt[3] + dem_width * dem_gt[4] + dem_height * dem_gt[5]
            dem_maxx = dem_gt[0] + dem_width * dem_gt[1] + dem_height * dem_gt[2]
            dem_maxy = dem_gt[3]
            dem_x = np.arange(dem_width) * dem_gt[1] + dem_gt[0]
            dem_y = np.arange(dem_height) * dem_gt[5] + dem_gt[3]
            # in the later code, I need to compare different positions with each other in order to determine the
            # position of a tile relative to another tile. With the above calculations, position that should be same
            # are differing at around 13 decimals, therefore the check for equality fails even though it should work
            dem_x = np.array([np.around(xi, 13) for xi in dem_x])
            dem_y = np.array([np.around(yi, 13) for yi in dem_y])
            if dem_X is None:
                dem_X, dem_Y = np.meshgrid(dem_x, dem_y)
                dem_val = np.array([dem_data.GetRasterBand(1).ReadAsArray()])
            else:
                dem_X_to_append, dem_Y_to_append = np.meshgrid(dem_x, dem_y)
                dem_val_to_append = np.array([dem_data.GetRasterBand(1).ReadAsArray()])
                dem_X, dem_Y, dem_val = dem_append(dem_X, dem_Y, dem_val, dem_X_to_append, dem_Y_to_append,
                                                        dem_val_to_append)
                write_result_tif_custom_ref(dem_val, out_path_merged, dem_datas[0], dem_X, dem_Y) # TODO remove this line after debugging
        write_result_tif_custom_ref(dem_val, out_path_merged, dem_datas[0], dem_X, dem_Y)
        result_dem = interpolate_merged_dem(out_path_merged, label_path)
        write_result_tif(result_dem, out_path, label_data)
        print('successfully wrote ' + out_path)


def get_dem_tif_srtm(tar_path, hand_path, res_out_path):
    print('Align DEM data with sen1floods11')
    # generate a tif file corresponding to the label lat/lon values with the DEM values
    dirs = glob(os.path.join(tar_path, "*", ""))
    # iterate through all sen1floods11 location folders
    for dir in dirs:
        print(f'Working on directory {dir}')
        files = os.listdir(dir)
        dem_files = list()
        # iterate through all files inside a location
        for file in files:
            # This is the convention behind our naming of downloaded SRTM files
            if file == 'DEM.tif':
                file_path = os.path.join(dir, file)
                dem_files.append(file_path)
                os_slash = get_os_slash()
                # need this if because folder names in labeled_source and labeled_label have different names and only
                # one name matches with the actual label file. Therefore change the name to the correct naming
                location_folder = file_path.split(os_slash)[-2].split('_')
                if location_folder[-1] == 'label':
                    location_folder = '_'.join(location_folder[:-1])
                else:
                    location_folder = '_'.join(location_folder)
                # generate path to corresponding label file
                label_folder_name = hand_path.split(get_os_slash())[-1]
                label_path = os.path.join(hand_path, location_folder + '_' + label_folder_name + '.tif')
                # generate output file name
                file_name = file_path.split(os_slash)[-2]
                out_path_short = os.path.join(res_out_path, file_name)
                out_path = os.path.join(out_path_short, file_name + '_dem.tif')
                # out_path_merged = os.path.join(out_path_short, file_name + 'total_merged_dem.tif')
                if not os.path.exists(out_path_short):
                    os.makedirs(out_path_short)
                # label_path = os.path.join(hand_path, 'Bolivia_103757_LabelHand.tif')

        # read dem and label data
        label_data = gdal.Open(label_path, gdal.GA_ReadOnly)
        result_dem = interpolate_srtm_dem(dem_files[0], label_path)
        write_result_tif(result_dem, out_path, label_data)
        print('successfully wrote ' + out_path)


def get_essential_info_from_tif(tif_path):
    tif_data = gdal.Open(tif_path, gdal.GA_ReadOnly)
    tif_width = tif_data.RasterXSize
    tif_height = tif_data.RasterYSize
    tif_gt = tif_data.GetGeoTransform()
    tif_minx = tif_gt[0]
    tif_miny = tif_gt[3] + tif_width * tif_gt[4] + tif_height * tif_gt[5]
    tif_maxx = tif_gt[0] + tif_width * tif_gt[1] + tif_height * tif_gt[2]
    tif_maxy = tif_gt[3]
    tif_x = np.arange(tif_width) * tif_gt[1] + tif_gt[0]
    tif_y = np.arange(tif_height) * tif_gt[5] + tif_gt[3]
    tif_X, tif_Y = np.meshgrid(tif_x, tif_y)
    tif_val = np.array([tif_data.GetRasterBand(1).ReadAsArray()])
    return tif_X, tif_Y, tif_val


def interpolate_srtm_dem(dem_path, target_tif_path):
    merged_dem_X, merged_dem_Y, merged_dem_val = get_essential_info_from_tif(dem_path)
    grid_x, grid_y, _ = get_essential_info_from_tif(target_tif_path)
    points, values = generate_point_value_pair(merged_dem_X, merged_dem_Y, merged_dem_val[0], grid_x, grid_y)
    interpolated_dem = interpolate.griddata(points, values, (grid_x, grid_y), method='cubic')
    # return interpolate_dem as such an array with an additional dimension, because that is the shape which is
    # returned by reading the value of a band with gdal. Just being consistent with other methods!
    return np.array([interpolated_dem])


def interpolate_merged_dem(dem_path, target_tif_path):
    merged_dem_X, merged_dem_Y, merged_dem_val = get_essential_info_from_tif(dem_path)
    grid_x, grid_y, _ = get_essential_info_from_tif(target_tif_path)
    points, values = generate_point_value_pair(merged_dem_X, merged_dem_Y, merged_dem_val[0], grid_x, grid_y)
    interpolated_dem = interpolate.griddata(points, values, (grid_x, grid_y), method='cubic')
    # return interpolate_dem as such an array with an additional dimension, because that is the shape which is
    # returned by reading the value of a band with gdal. Just being consistent with other methods!
    return np.array([interpolated_dem])


def generate_point_value_pair(grid_X, grid_Y, val, ref_x, ref_y):
    # generate a list of x,y points corresponding to a list of values. Will remove all points not inside the reference
    # coordinate frame
    # need grid_X, grid_Y and val need to have the same shapes
    assert grid_X.shape == grid_Y.shape and grid_X.shape == val.shape, 'grid_X, grid_Y and val need to have the same ' \
                                                                       'shapes '
    y_elems = grid_X.shape[0]
    x_elems = grid_X.shape[1]
    points = np.zeros((y_elems * x_elems, 2))
    values = np.zeros((y_elems * x_elems))
    i = 0
    indices_to_remove = list()
    epsilon_x = abs(ref_x[0, 0] - ref_x[0, 1]) * 3
    epsilon_y = abs(ref_y[0, 0] - ref_y[1, 0]) * 3
    x_range = [ref_x[0, 0], ref_x[0, -1]]
    y_range = [ref_y[0, 0], ref_y[-1, 0]]
    x_range.sort()
    y_range.sort()
    x_range[0] = x_range[0] - epsilon_x
    x_range[1] = x_range[1] + epsilon_x
    y_range[0] = y_range[0] - epsilon_y
    y_range[1] = y_range[1] + epsilon_y
    # TODO: this can be improved to be faster
    for y in range(y_elems):
        for x in range(x_elems):
            points[i] = np.array([grid_X[y, x], grid_Y[y, x]])
            values[i] = val[y, x]
            if (grid_X[y, x] <= x_range[0]) or (grid_X[y, x] >= x_range[1]):
                indices_to_remove.append(i)
                # we only want to add the index once, so if it has been already added, we already know the point is
                # outside the reference frame and we can skip the second if
            elif (grid_Y[y, x] <= y_range[0]) or (grid_Y[y, x] >= y_range[1]):
                indices_to_remove.append(i)
            i += 1
    indices_to_remove = np.array(indices_to_remove)
    points = np.delete(points, indices_to_remove, axis=0)
    values = np.delete(values, indices_to_remove, axis=0)
    return points, values


def check_location(X, X_to_append, Y, Y_to_append):
    # Investigate if X_to_append is before, after or same X-level and investigate
    # if Y_to_append is above, below and the same Y-level.
    # returns two ints which denote the direction for each x and y-axis
    if (X == X_to_append).all():
        # print('Same X-level')
        x_dir = 0
    elif X[0, 0] == X_to_append[0, -1]:
        # print('before')
        x_dir = -1
    elif X[0, -1] == X_to_append[0, 0]:
        # print('after')
        x_dir = 1
    else:
        sys.exit('Not all cases defined')

    if (Y == Y_to_append).all():
        # print('Same Y-level')
        y_dir = 0
    elif Y[0, 0] == Y_to_append[-1, 0]:
        # print('above')
        y_dir = -1
    elif Y[-1, 0] == Y_to_append[0, 0]:
        # print('below')
        y_dir = 1
    else:
        sys.exit('Not all cases defined')
    return x_dir, y_dir


def append_maps(X, Y, val, X_to_append, Y_to_append, val_to_append, x_dir, y_dir):
    # append the additional grid at the appropriate location.
    # first concatenated element excludes last list entry because it is overlapping!
    # if either x_dir or y_dir is 0, then either X or Y values stay the same and DEM values only need to be appended
    # once in one direction. Therefore, first if catches the case that we have to append left, right, top or bottom
    # of initial tile
    if not (x_dir != 0 and y_dir != 0):  # make sure both are not unqeual zero
        if x_dir == 0:
            return_X = X[0, :]
        elif x_dir == -1:
            return_X = np.concatenate((X_to_append[0, :-1], X[0, :]))
            return_dem = np.array([np.hstack((val_to_append[0][:, :-1], val[0]))])
        elif x_dir == 1:
            return_X = np.concatenate((X[0, :-1], X_to_append[0, :]))
            return_dem = np.array([np.hstack((val[0][:, :-1], val_to_append[0]))])

        if y_dir == 0:
            return_Y = Y[:, 0]
        elif y_dir == -1:
            return_Y = np.concatenate((Y_to_append[:-1, 0], Y[:, 0]))
            return_dem = np.array([np.vstack((val_to_append[0][:-1, :], val[0]))])
        elif y_dir == 1:
            return_Y = np.concatenate((Y[:-1, 0], Y_to_append[:, 0]))
            return_dem = np.array([np.vstack((val[0][:-1, :], val_to_append[0]))])

        return_X, return_Y = np.meshgrid(return_X, return_Y)

    # in this case we have to append on a corner not a border. Meaning, our second tile is either above/left,
    # above/right, below/left or below/right
    elif x_dir != 0 and y_dir != 0:
        filler1 = np.full(val_to_append.shape, -9999)
        filler2 = np.full(val_to_append.shape, -9999)
        if y_dir == -1 and x_dir == -1:
            # print('above and before')
            return_Y = np.concatenate((Y_to_append[:-1, 0], Y[:, 0]))
            return_X = np.concatenate((X_to_append[0, :-1], X[0, :]))
            # add above
            dem_init = np.array(np.vstack((filler1[0][:-1, :], val[0])))
            # add below
            dem_to_append = np.array(np.vstack((val_to_append[0], filler2[0][:-1, :])))
            # combine partial DEMs
            return_dem = np.array([np.hstack((dem_to_append[:, :-1], dem_init))])

        if y_dir == 1 and x_dir == -1:
            # print('below and before')
            return_Y = np.concatenate((Y[:-1, 0], Y_to_append[:, 0]))
            return_X = np.concatenate((X_to_append[0, :-1], X[0, :]))
            # add above
            dem_to_append = np.array(np.vstack((filler1[0][:-1, :], val_to_append[0])))
            # add below
            dem_init = np.array(np.vstack((val[0], filler2[0][:-1, :])))
            # combine partial DEMs
            return_dem = np.array([np.hstack((dem_to_append[:, :-1], dem_init))])

        if y_dir == -1 and x_dir == 1:
            # print('above and after')
            return_Y = np.concatenate((Y_to_append[:-1, 0], Y[:, 0]))
            return_X = np.concatenate((X[0, :-1], X_to_append[0, :]))
            # add above
            dem_init = np.array(np.vstack((filler1[0][:-1, :], val[0])))
            # add below
            dem_to_append = np.array(np.vstack((val_to_append[0], filler2[0][:-1, :])))
            # combine partial DEMs
            return_dem = np.array([np.hstack((dem_init[:, :-1], dem_to_append))])

        if y_dir == 1 and x_dir == 1:
            # print('below and after')
            return_Y = np.concatenate((Y[:-1, 0], Y_to_append[:, 0]))
            return_X = np.concatenate((X[0, :-1], X_to_append[0, :]))
            # add above
            dem_to_append = np.array(np.vstack((filler1[0][:-1, :], val_to_append[0])))
            # add below
            dem_init = np.array(np.vstack((val[0], filler2[0][:-1, :])))
            # combine partial DEMs
            return_dem = np.array([np.hstack((dem_init[:, :-1], dem_to_append))])

        return_X, return_Y = np.meshgrid(return_X, return_Y)
    else:
        sys.exit('Missed case in append_maps')
    return return_X, return_Y, return_dem


def dem_append(X, Y, val, X_to_append, Y_to_append, val_to_append):
    # check if X_to_append and Y_to_append grid is to be added before/after/below/above or at the same position
    # relative to original X,Y and then append it at the found location.

    # this if catches that we try to append DEM maps of the same size, meaning the first appending step
    if X.shape == X_to_append.shape and Y.shape == Y_to_append.shape and val.shape == val_to_append.shape:
        x_dir, y_dir = check_location(X, X_to_append, Y, Y_to_append)
        return_X, return_Y, return_dem = append_maps(X, Y, val, X_to_append, Y_to_append, val_to_append, x_dir, y_dir)
    # catch the case that we append to an already merged DEM map
    else:
        # TODO: the following method will fail if we are not in the sen1floods11 case. In other cases it might be added
        # below and in the middle of a three tile row. x_dir would then need to be extended to represent the position
        # itself and not just before or after. filler_dem would then need to be created with respect to that. E.g.
        # around it and not just before or after. Check location and the above appending methods could be reused when
        # we append to a long row or column. But this will not happen in sen1floods11
        x_dir, y_dir = check_location_diff_shape(X, X_to_append, Y, Y_to_append)
        return_X, return_Y, return_dem = append_maps_diff_shape(X, Y, val, X_to_append, Y_to_append, val_to_append,
                                                                x_dir, y_dir)

    # print(return_X.shape, return_Y.shape, return_dem.shape)
    return return_X, return_Y, return_dem


def append_maps_diff_shape(X, Y, val, X_to_append, Y_to_append, val_to_append, x_dir, y_dir):
    # similar to append_maps
    if x_dir == -3 or y_dir == -3:  # both cannot be true in the case with unequal shapes
        # append on a row of tiles in x or y-direction another tile
        if x_dir == -3:  # same x-level
            return_X = X[0, :]
        elif x_dir == -1:  # before
            return_X = np.concatenate((X_to_append[0, :-1], X[0, :]))
            return_dem = np.array([np.hstack((val_to_append[0][:, :-1], val[0]))])
        elif x_dir == -2:  # after
            return_X = np.concatenate((X[0, :-1], X_to_append[0, :]))
            return_dem = np.array([np.hstack((val[0][:, :-1], val_to_append[0]))])
        else:
            sys.exit('Forgot case 1')

        if y_dir == -3:  # same y-level
            return_Y = Y[:, 0]
        elif y_dir == -1:  # above
            return_Y = np.concatenate((Y_to_append[:-1, 0], Y[:, 0]))
            return_dem = np.array([np.vstack((val_to_append[0][:-1, :], val[0]))])
        elif y_dir == -2:  # below
            return_Y = np.concatenate((Y[:-1, 0], Y_to_append[:, 0]))
            return_dem = np.array([np.vstack((val[0][:-1, :], val_to_append[0]))])
        else:
            sys.exit('Forgot case 2')

        return_X, return_Y = np.meshgrid(return_X, return_Y)

    # we have a row or column of two tile in sen1floods11, then we want to add a third to a specific point at the left
    # or right. Or we want to add a third one to a specific point above or below. The appended tile needs to be extended
    # with filler to the appropriate dimension
    elif x_dir != -3 and y_dir != -3:
        filler = np.full(val_to_append.shape, -9999)
        if y_dir == -1 and x_dir == -1:
            sys.exit(f'x_dir={x_dir} and y_dir={y_dir} cannot happen in sen1floods11 in append_maps_diff_shape')
            # print('above and before')
        elif y_dir == -2 and x_dir == -1:
            sys.exit(f'x_dir={x_dir} and y_dir={y_dir} cannot happen in sen1floods11 in append_maps_diff_shape')
            # print('below and before')
        elif y_dir == -1 and x_dir == -2:
            sys.exit(f'x_dir={x_dir} and y_dir={y_dir} cannot happen in sen1floods11 in append_maps_diff_shape')
            # print('above and after')
        elif y_dir == -2 and x_dir == -2:
            sys.exit(f'x_dir={x_dir} and y_dir={y_dir} cannot happen in sen1floods11 in append_maps_diff_shape')
            # print('below and after')
        elif y_dir == -2 and x_dir >= 0:
            # print('below and at a specific x-position')
            # generate the new x and y lists
            return_Y = np.concatenate((Y[:-1, 0], Y_to_append[:, 0]))
            return_X = X[0, :]
            num_steps = (return_X.shape[0] - 1)/(X_to_append[0, :].shape[0]-1)
            # I want x_dir many fillers appended before x_dir and (num_steps-1)-x_dir appended afterwards
            if x_dir > 0:
                dem_to_append_before = np.tile(filler[0], (1, x_dir))  # TODO: for sen1floods11 this is fine, might be an error in general cases where we have more than 2 tiles in a row, because each tile has len 3601 but we only add 3600. Check this later
            else:
                dem_to_append_before = None
            if ((num_steps-1)-x_dir) > 0:
                dem_to_append_after = np.tile(filler[0], (1, ((num_steps-1)-x_dir)))
            else:
                dem_to_append_after = None

            # append the fillers to the appending values
            if dem_to_append_before is not None:
                dem_to_append = np.array(np.hstack((dem_to_append_before[:, :-1], val_to_append[0])))
            else:
                dem_to_append = val_to_append[0].copy()
            if dem_to_append_after is not None:
                dem_to_append = np.array(np.hstack((dem_to_append, dem_to_append_after[:, 1:])))
            #else:
            #    sys.exit('Should not get in this case1')

            # combine partial DEMs
            # change logic from append_maps_diff because there it didn't matter from which map we remove one row
            # now it matters. We want to keep the one with complete values. The one which gets appended contains filler
            # values. So it is preferable to discard them.
            return_dem = np.array([np.vstack((val[0], dem_to_append[1:, :]))])

        elif y_dir == -1 and x_dir >= 0:
            # print('above and at a specific x-position')
            # generate the new x and y lists
            return_Y = np.concatenate((Y_to_append[:-1, 0], Y[:, 0]))
            return_X = X[0, :]
            num_steps = (return_X.shape[0] - 1) / (X_to_append[0, :].shape[0] - 1)
            # I want x_dir many fillers appended before x_dir and (num_steps-1)-x_dir appended afterwards
            if x_dir > 0:
                dem_to_append_before = np.tile(filler[0], (1, x_dir))
            else:
                dem_to_append_before = None
            if ((num_steps - 1) - x_dir) > 0:
                dem_to_append_after = np.tile(filler[0], (1, ((num_steps - 1) - x_dir)))
            else:
                dem_to_append_after = None

            # append the fillers to the appending values
            if dem_to_append_before is not None:
                dem_to_append = np.array(np.hstack((dem_to_append_before[:, :-1], val_to_append[0])))
            else:
                dem_to_append = val_to_append[0].copy()
            if dem_to_append_after is not None:
                dem_to_append = np.array(np.hstack((dem_to_append, dem_to_append_after[:, 1:])))
            #else:
            #    sys.exit('Should not get in this case2')

            # combine partial DEMs
            # change logic from append_maps_diff because there it didn't matter from which map we remove one row
            # now it matters. We want to keep the one with complete values. The one which gets appended contains filler
            # values. So it is preferable to discard them.
            return_dem = np.array([np.vstack((dem_to_append[:-1, :], val[0]))])

        elif x_dir == -1 and y_dir >= 0:
            # print('before and at a specific y-location')
            # generate the new x and y lists
            return_Y = Y[:, 0]
            return_X = np.concatenate((X_to_append[0, :-1], X[0, :]))
            num_steps = (return_Y.shape[0] - 1) / (Y_to_append[:, 0].shape[0] - 1)
            # I want y_dir many fillers appended below y_dir and (num_steps-1)-y_dir appended above
            if y_dir > 0:
                dem_to_append_below = np.tile(filler[0], (y_dir, 1))
            else:
                dem_to_append_below = None
            if ((num_steps - 1) - y_dir) > 0:
                dem_to_append_above = np.tile(filler[0], (((num_steps - 1) - y_dir), 1))
            else:
                dem_to_append_above = None

            # append the fillers to the appending values
            if dem_to_append_below is not None:
                dem_to_append = np.array(np.vstack((val_to_append[0], dem_to_append_below[1:, :])))
            else:
                dem_to_append = val_to_append[0].copy()
            if dem_to_append_above is not None:
                dem_to_append = np.array(np.vstack((dem_to_append_above[:-1, :], dem_to_append)))
            #else:
            #    sys.exit('Should not get in this case3')

            # combine partial DEMs
            # change logic from append_maps_diff because there it didn't matter from which map we remove one row
            # now it matters. We want to keep the one with complete values. The one which gets appended contains filler
            # values. So it is preferable to discard them.
            return_dem = np.array([np.hstack((dem_to_append[:, :-1], val[0]))])
        elif x_dir == -2 and y_dir >= 0:
            # print('after and at a specific y-location')
            # generate the new x and y lists
            return_Y = Y[:, 0]
            return_X = np.concatenate((X[0, :-1], X_to_append[0, :]))
            num_steps = (return_Y.shape[0] - 1) / (Y_to_append[:, 0].shape[0] - 1)
            # I want y_dir many fillers appended below y_dir and (num_steps-1)-y_dir appended above
            if y_dir > 0:
                dem_to_append_below = np.tile(filler[0], (y_dir, 1))
            else:
                dem_to_append_below = None
            if ((num_steps - 1) - y_dir) > 0:
                dem_to_append_above = np.tile(filler[0], (((num_steps - 1) - y_dir), 1))
            else:
                dem_to_append_above = None

            # append the fillers to the appending values
            if dem_to_append_below is not None:
                dem_to_append = np.array(np.vstack((val_to_append[0], dem_to_append_below[1:, :])))
            else:
                dem_to_append = val_to_append[0].copy()
            if dem_to_append_above is not None:
                dem_to_append = np.array(np.vstack((dem_to_append_above[:-1, :], dem_to_append)))
            #else:
            #    sys.exit('Should not get in this case4')

            # combine partial DEMs
            # change logic from append_maps_diff because there it didn't matter from which map we remove one row
            # now it matters. We want to keep the one with complete values. The one which gets appended contains filler
            # values. So it is preferable to discard them.
            return_dem = np.array([np.hstack((val[0], dem_to_append[:, 1:]))])
        elif x_dir >= 0 and y_dir >= 0:
            # in this case we don't want to append a tile, but actually overwrite filler values added by a previous
            # append operation. Therefore, x_dir and y_dir define a location inside the currently merged DEM
            # print('we need to overwrite')
            # grid system stays the same
            return_Y = Y[:, 0]
            return_X = X[0, :]
            x_len = val_to_append[0].shape[1]
            y_len = val_to_append[0].shape[0]
            # overwrite at appropriate location. x_dir==0 and y_dir==0 is the lower left corner
            # print(val[0][(y_len-1)*y_dir:((y_len-1) * (y_dir + 1) + 1), (x_len-1)*x_dir:((x_len-1) * (x_dir + 1) + 1)].shape)
            # big_list[((length - 1) * i):((length - 1) * (i + 1) + 1)]
            val[0][(y_len-1)*y_dir:((y_len-1) * (y_dir + 1) + 1), (x_len-1)*x_dir:((x_len-1) * (x_dir + 1) + 1)] = val_to_append[0]
            return_dem = val.copy()
        else:
            sys.exit('Forgot a case in append_maps_diff_shape')
        return_X, return_Y = np.meshgrid(return_X, return_Y)
    else:
        sys.exit('Missed case in append_maps_diff_shape')
    return return_X, return_Y, return_dem


def check_location_diff_shape(X, X_to_append, Y, Y_to_append):
    # similar to check_location but works with different shapes of reference and to appended matrix
    if X[0, :].shape == X_to_append[0, :].shape:
        if (X[0, :] == X_to_append[0, :]).all():
            # print('Same X-level')
            x_dir = -3  # 0
        elif X[0, 0] == X_to_append[0, -1]:
            # print('before')
            x_dir = -1
        elif X[0, -1] == X_to_append[0, 0]:
            # print('after')
            x_dir = -2  # 1
        else:
            sys.exit('Not all cases defined')
    elif is_inside_bool(X[0, :], X_to_append[0, :]):
        x_dir = is_inside_index(X[0, :], X_to_append[0, :])
        # print('Is at the x-level given by x_dir')
    else:
        sys.exit('Not all cases defined')

    if Y[:, 0].shape == Y_to_append[:, 0].shape:
        if (Y[:, 0] == Y_to_append[:, 0]).all():
            # print('Same Y-level')
            y_dir = -3
        elif Y[0, 0] == Y_to_append[-1, 0]:
            # print('above')
            y_dir = -1
        elif Y[-1, 0] == Y_to_append[0, 0]:
            # print('below')
            y_dir = -2
        else:
            sys.exit('Not all cases defined1')
    elif is_inside_bool(Y[:, 0], Y_to_append[:, 0]):
        y_dir = is_inside_index(Y[:, 0], Y_to_append[:, 0])
        # print('Is at the y-level given by y_dir')
    else:
        sys.exit('Not all cases defined')
    return x_dir, y_dir


def is_inside_bool(big_list, small_list):
    # check if small list is inside big_list. Sublists of big_list are of size small_list.shape and all sublists
    # are not overlapping
    i = 0
    length = small_list.shape[0]  # 3601 and big_list.shape[0] is e.g. 7201
    # need -1 because one tile has 3601 elements. Due to an overlap of one row/column, however we only add
    # 3600 elements in an iteration. Therefore, the division needs that in order to work out. This is also the reason
    # for the unnatural list slicing position of ((length-1)*(i+1)+1)
    while i < ((big_list.shape[0]-1)/(length-1)):
        if (big_list[((length-1)*i):((length-1)*(i+1)+1)] == small_list).all():
            return True
        else:
            i += 1
    return False


def is_inside_index(big_list, small_list):
    # similar to is_inside_bool, but it returns the index postion instead of a bool
    i = 0
    length = small_list.shape[0]
    # need -1 because one tile has 3601 elements. Due to an overlap of one row/column, however we only add
    # 3600 elements in an iteration. Therefore, the division would be slightly lower. This is also the reason for the
    # unnatural list slicing position of ((length-1)*(i+1)+1)
    while i < ((big_list.shape[0]-1)/(length-1)):
        # adder = 0 if i == 0 else 1
        # if (big_list[((length-1)*i+adder):((length-1)*(i+1)+1)] == small_list).all():
        if (big_list[((length - 1) * i):((length - 1) * (i + 1) + 1)] == small_list).all():
            return i
        else:
            i += 1
    # at one point we will return i, otherwise is_inside_bool would have returned False in the previous step
    sys.exit('My assumption in is_inside_index is wrong. Probably did a mistage in the while loop condition of '
             'is_inside_bool')


def write_result_tif(result_dem, out_path, ref_data):
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = result_dem.shape[1], result_dem.shape[2]
    outdata = driver.Create(out_path, cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ref_data.GetGeoTransform())  # sets same geotransform as input
    outdata.SetProjection(ref_data.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(result_dem[0])
    outdata.GetRasterBand(1).SetNoDataValue(-9999)  # if you want these values transparent
    outdata.FlushCache()  # saves to disk!!
    outdata = None


def write_result_tif_custom_ref(result_dem, out_path, ref_data, dem_X, dem_Y):
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = result_dem.shape[1], result_dem.shape[2]
    outdata = driver.Create(out_path, cols, rows, 1, gdal.GDT_UInt16)
    geoprojection = generateGeoTransform(dem_X, dem_Y)
    outdata.SetGeoTransform(geoprojection)  # sets same geotransform as input
    outdata.SetProjection(ref_data.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(result_dem[0])
    outdata.GetRasterBand(1).SetNoDataValue(-9999)  # if you want these values transparent
    outdata.FlushCache()  # saves to disk!!
    outdata = None


def generateGeoTransform(dem_X, dem_Y):
    # https://gdal.org/tutorials/raster_api_tut.html under section "getting Dataset information"
    # https://gdal.org/tutorials/geotransforms_tut.html
    top_left_x = dem_X[0, 0]  # smalles x-value should be the very first value
    w_e_resolution = dem_X[0, 1] - dem_X[
        0, 0]  # order is important! absolute shouldn't be used, because resolution can be negative
    row_rotation = 0
    top_left_y = dem_Y[0, 0]  # largest y-value should be the very first value
    column_rotation = 0
    n_s_resolution = dem_Y[1, 0] - dem_Y[0, 0]
    # (-65.0001388888889, 0.000277777777777778, 0.0, -13.9998611111111, 0.0, -0.000277777777777778)
    return (top_left_x, w_e_resolution, row_rotation, top_left_y, column_rotation, n_s_resolution)


def get_closest_dem_index(x_axis, y_axis, lon, lat):
    list_ind = get_list_ind(x_axis, y_axis, lon, lat)
    x_ind = find_closest_index(x_axis[list_ind][0], lon)
    y_ind = find_closest_index(y_axis[list_ind][:, 0], lat)
    # print(x_axis[0][x_ind], y_axis[:,0][y_ind])
    return x_ind, y_ind, list_ind


def get_list_ind(x_axis, y_axis, lon, lat):
    # iterates through a list of x and y axis values and finds the index of the list element (grid tile) which
    # contains the given lon and lat values
    for idx, (x, y) in enumerate(zip(x_axis, y_axis)):
        x_hit = False
        y_hit = False
        x_list = x[0]
        y_list = y[:, 0]
        x_hit = check_is_inside(x_list, lon)
        y_hit = check_is_inside(y_list, lat)
        if x_hit and y_hit:
            return idx
    sys.exit('This method fails')


def check_is_inside(elements, value):
    # checks if a given value is inside an array elements
    # elements is sorted, but we don't know if it is ascending or descending
    if elements[0] == value:
        return True
    elif elements[-1] == value:
        return True
    # first element might be largest or smallest and the last element is the corresponding other
    elif elements[0] > value:
        if elements[-1] < value:
            return True
        else:
            return False
    elif elements[0] < value:
        if elements[-1] > value:
            return True
        else:
            return False
    else:
        sys.exit('This method did not cover all possible cases!!!')


def find_closest_index(coord_list, coord):
    # find the index in an array whose value is closest to a given value
    val = min(coord_list, key=lambda x: abs(x - coord))
    return np.where(coord_list == val)[0][0]


def get_os_slash():
    return os.path.join('a', 'a')[1]


def parse_arugments():

    parser = argparse.ArgumentParser(description=
                                     'This script downloads the DEM filess according to the tile information of the '
                                     'input folder. Works for sen1floods11 dataset. Note that input and reference '
                                     'folder must refer to the same kind of data. E.g. both should refer to hand '
                                     'labeled or weakly labeled data.')
    parser.add_argument('-i', '--in_folder', dest='in_folder', type=str, required=True,
                        help='Input folder path to where the collection.json is defined. E.g. '
                             'data/sen1floods11_data_toy/v1.1/catalog/sen1floods11_hand_labeled_label')
    parser.add_argument('-d', '--download_folder', dest='download_folder', type=str, required=True,
                        help='Folder path to where the results should be downloaded')
    parser.add_argument('-o', '--out_folder', dest='out_folder', type=str, required=True,
                        help='Folder path to where the results (final DEM map) should be stored')
    parser.add_argument('-r', '--reference_folder', dest='ref_folder', type=str, required=True,
                        help='Folder path to where the label information is stored. In the case of hand labeled '
                             'data, it must lead to the LabelHand folder and in the case of weakly labeled data it '
                             'must lead to the S1OtsuLabelWeak folder')
    parser.add_argument('-s', '--source', dest='source', type=str, required=True, help='Define which datasource should '
                                                                                       'be used. Available options are:'
                                                                                       '\'SRTM\' and \'ASTER\'')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arugments()
    if os.path.exists(args.download_folder):
        sys.exit(f'Download folder {args.download_folder} already exists')
    if os.path.exists(args.out_folder):
        sys.exit(f'Output folder {args.out_folder} already exists')
        #shutil.rmtree(args.out_folder)
        #os.makedirs(args.out_folder)
    if args.source == 'ASTER':
        get_DEM(args.in_folder, args.download_folder)
        get_dem_tif_aster(args.download_folder, args.ref_folder, args.out_folder)
    elif args.source == 'SRTM':
        get_DEM_SRTM(args.in_folder, args.download_folder)
        get_dem_tif_srtm(args.download_folder, args.ref_folder, args.out_folder)
    else:
        sys.exit('Invalid source argument. Please call -s with \'SRTM\' or \'ASTER\'')
