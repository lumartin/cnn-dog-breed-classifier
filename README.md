# Dog Breed Classification

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This repository contains code supporting the process of building a dog breed classification system from scratch, including the Data Science and Software Engineering tasks needed for the project.

  - Benchmark notebook: [dog_app.ipynb](https://github.com/lumartin/cnn-dog-breed-classifier/blob/master/dog_app.ipynb)
  - Exploratory Data Analysis: [Analysis](https://github.com/lumartin/cnn-dog-breed-classifier/tree/master/analysis)
  - Experiments with different hyperparameters and architectures: [Experiments](https://github.com/lumartin/cnn-dog-breed-classifier/tree/master/experiments)
  - Utility framework for experimentation: [Utils](https://github.com/lumartin/cnn-dog-breed-classifier/tree/master/utils)

## Machine Learning techniques

Given that this is an image classification problem, I have decided to use Convolutional Neural Networks for the solution, as they are the state-of-the-art technology for this matter. 

I have explored several ways to approach the problem, divided in two main groups: from-scratch implementation and transfer learning implementation. This type of problem is clearly better solved using a transfer learning approach, but I think it worth it to make some research in the other approach. Most of the work I have made for this project has been in this context. 

The main technology used has been [PyTorch](https://pytorch.org/), the Facebook Neural Networks Library. It gives me the ability to build and train neural networks from scratch, as well as use common transfer learning solutions. Together with PyTorch, I have made use of common Python libraries like Numpy and Matplotlib. [Imgaug](https://imgaug.readthedocs.io/en/latest/) is also used as augmentation library. 


## Technology

Most of the notebooks included in the Experiments folder are configured to work in Google Colab with Google Drive integrated, but most of the code is agnostic of the technology used. Of course I suggest using a Nvidia GPU for training.

The code you can find inside the utils folder is an experimentation framework developed for PyTorch. This library exposes a single method called “run_experiments”, that receives a dictionary of hyperparameters and runs experiments making combinations of the values received. For example, we can specify that we want a set of experiments that combine two different optimizers, such as Adam  and Adagrad , and two learning rates, say 0.01 and 0.03. With these settings, the framework will perform the following experiments:
- Optimizer: Adam, Learning Rate: 0.01
- Optimizer: Adam, Learning Rate: 0.03
- Optimizer: Adagrad, Learning Rate: 0.01
- Optimizer: Adagrad, Learning Rate: 0.03

More concretely, the allowed hyperparameters that can be included in the configuration are the following:
- **augmentations**: Specify a list of imgaug augmentation methods.
- **learning_rates**: List of learning rate values
- **epochs**: Number of training epochs
- **optimizers**: List of optimizers
- **models**: List of models

This is the signature for the “run_experiments” function:
```
def run_experiments(paths, hyperparameters, model_file='model')
```
where paths is a dictionary specifying, hyperparameters is the dictionary described previously, and model_file is the name of the file that we want for storing our model. 
Everything else is hidden (preprocessing, training and testing), but we can still use the generated model to make calculations. 

