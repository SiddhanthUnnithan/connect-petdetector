"""
    MobileNet training script

    Includes data download, initial training, 
    datastore creation for hyperparameter tuning and
    model outputting.

"""
from scripts.retrain import train
from scripts.oxford_dataset_helpers import fetch_and_untar, move_images_into_labelled_directories

# need to download and transform training data
# ideal - run this as a separate pipeline step and push into a datastore
fetch_and_untar('http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz')
move_images_into_labelled_directories("images")

# initial training run
train(architecture='mobilenet_0.50_224', 
         image_dir='images',
         output_dir='outputs', 
         bottleneck_dir='bottleneck',
         model_dir='model',
         learning_rate=0.00008, 
         training_steps=500,         
         use_hyperdrive=False)
