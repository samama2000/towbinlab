# towbinlab
Variety of Image Analysis and Data Handling Tools for the Towbin Lab at the University of Bern

## channel_separator

Use this to generate single-channel TIFF files for micro_sam annotation.


## conversion

Use this to convert SQUID imaging BMP-files into three-channel TIFF images.


# models

Contains the CSRNet egg counting and U-Net++ body segmentation models. The body segmentation model was too large to upload as a single file. Instructions to retrieve the model are given in the directory. 


## csrnet

Use this to count eggs in brightfield image channels.


# stardist

Use this to count eggs in body fluorescent image channels.


# data_analysis

Use this to load and merge multiple experiment filemaps with egg counts and make the resulting data analysis on growth and reproduction rates (for mex3.ipynb) as well as lifespan (with lifespan.ipynb).


# plotting

Contains a variety of useful plotting functions e.g.
- plotting training metrics for towbintools pipeline training
- plotting all growth functions for each worm
- plotting all egg counts for each worm
- generate a gif for each worm from channel images or channel masks

