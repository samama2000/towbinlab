To create a training set to train an xgboost model to give semantic meaning to the stardist instance segmentation (ie. which are egg-like and which are worm-like)
I did the following:

1. chose a sample of n raw images from the experiment's dataset and extracted channel 2 from them
    - 'choosing_training_data.ipynb'
2. annotated the worm within the sample using microsam and napari on my local laptop (not on the server!)
    - installation instructions: https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation
    - install with gpu support, otherwise interactive segmentation will be rather slow
    - installing microsam with conda also installs napari
    - I annotated the image samples using the 'Image Series Annotated' from the microsam plugin within napari
        - https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#image-series-annotator
        - note, I recommend explicitly setting the 'embeddings save path' under advanced settings
        - this will do the heavy computation up front (for me ca. 10min for 200 images on my laptop gpu) and save the results
        - then, during interactive annotation, the actual prompt segmentation will be very quick, so you don't have to wait long during each image
        - I annotated using vit_l model, however try whichever, it doesn't matter as long as you get the segmentation you want
3. ran stardist on the images
    - 'run_stardist_on_training.ipynb'
    - make sure to run in a slurm job with a gpu
4. ran feature processing on the stardist instance segmentation
    - 'feature_processing.ipynb'
    - extracts features for each region on which the xgboost classifier training and inference is ran on
5. trained the classifier
    - 'classifier_training.ipynb'
    