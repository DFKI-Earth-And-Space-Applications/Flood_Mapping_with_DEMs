# Flood_Mapping_with_DEMs
Public repository of our IGARSS 2023 submission.

This work is based on the [sen1floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset.

## Usage

Please download the data from the [sen1floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset.

Note, that the weakly labeled Sentinel-2 data is only contained in the old `cnn_chips` bucket.

Use `generate_sen1_dem.py` to download the DEM files with respect to the sen1floods11 files.

### generate_sen1_dem.py

Automatically downloads DEM maps related to the sen1floods11 dataset, unpacks downloaded content and aligns the DEM map to sen1floods11. Aligning means, to use the same latitute and longitude values of the DEM map as in the sen1floods11 maps. The resolution from the DEM map is increased to match the resolution of sen1floods11 by using cubic interpolation.

In a console run the following after adapting the values to your system:
```
python generate_sen1_dem.py -i data/sen1floods11_data/v1.1/catalog/sen1floods11_hand_labeled_source -d data/downloaded -o data/final_out -r data/sen1floods11_data/v1.1/data/flood_events/HandLabeled/LabelHand -s ASTER
```

* -i: denotes the path to the folder which contains the collection.json and all the folders for each location. It doesn't matter whether it is the _label or _source folder.
* -d: Path to the folder where the downloaded content will be stored. Note, the script creates the folder and will stop if it already exists, in order to prevent overwriting old results.
* -r: Path to where the reference files are stored. E.g. the label data of flood events as stored in sen1floods11_data/v1.1/data/flood_events/HandLabeled/LabelHand. However, the other folders such as S1Hand or S2Hand should work fine as well. However, if your -i marks a hand labeled structure, then -r also needs to refer a hand labeled structure. Or both need to refer weak labeled structure.
* -o: Path to where the final results should be stored. Similar to -d, the folder given with this parameter will be created and it throws an error if it already exists.
* -s: Define from which source to download. Possible options are: "SRTM" or "ASTER".
* Parameters only accept one value for now, therefore, script needs to be run at least twice, once for hand labeled data and once for weak labeled data.
* Note, that you will have to create a .wgetrc file with "touch .wgetrc | chmod og-rw .wgetrc" and enter your credentials into the file. Guide is here: https://lpdaac.usgs.gov/resources/e-learning/how-access-lp-daac-data-command-line/ in Section "2. Download LP DAAC data from the command line with wget". Of course, you will also need to have to create an account on the page.

Afterwards you can use `flood_dem.py` to reproduce our results.

### flood_dem.py

You can use `flood_dem.py` with command line arguments. However, the recommended usage is with configuration files like:

```
python flood_dem.py -cf cfg_files/config.json
```

The settable parameters are:
* n_runs": Number of runs with different seeds that should be performed
* source": Which DEM source to use
* gradient": Calculate the gradients in meter or gradient space
* depression": Application of depression filter
* flow_direction": The used Flow metric: Possible options are listed here: https://richdem.readthedocs.io/en/latest/flow_metrics.html (D8, D4, etc.)
* run_id": Number to give the output folder
* channels": Which channels to use during training. These include Sentinel-1 and Sentinel-2 channels, as well as the indices NDWI, MNDWI, AWEI, AWEISH. Hue, Saturation and Value can also be applied if the HSV transformation is used.
* HSV_channels": Mark the three channels on which the HSV transformation should be applied.
* weakly_flag": Mark if weakly labeled data should be used in addition to hand labeled data for training.

What kind of values are to be given can be seen in the example configuration at `cfg_files/config.json`. The method `parse_arugments` in `flood_dem.py` provides additional insights.

Note, that a `0` means, that a channel/feature/pre-processing method is not used while `1` obviously means it will be used.

TODO: Requires libraries with versions.
