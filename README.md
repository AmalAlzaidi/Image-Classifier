# Image Classifier


Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this [dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below.

### Installation
- Python 3.5+
- Python libraries: numpy, pandas.
- Machine Learning Libraries: Torch, Siki-Learn
- Image Process Libraries: PIL
- Argument processing library: Argparse 

### command line application

- Train a new network on a data set with train.py

- Basic usage: python train.py data_directory
- Prints out training loss, validation loss, and validation accuracy as the network trains
- Options:
    - Set directory to save checkpoints:
    python train.py data_dir --save_dir save_directory
    - Choose architecture: python train.py data_dir --arch "vgg13"
    - Set hyperparameters: 
    python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
     - Use GPU for training: python train.py data_dir --gpu
- Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

  - Basic usage: python predict.py /path/to/image checkpoint
  - Options:
     - Return top KK most likely classes: python predict.py input checkpoint --top_k 3
     - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
     - Use GPU for inference: python predict.py input checkpoint --gpu
     
     
 ### Data
 - [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
 - cat_to_name.json : A dictionary mapping the integer encoded categories to the actual names of the flowers.
 
 ### Acknowledgements
  This project was completed as a part of the Udacity Data Scientist Nanodegree.
