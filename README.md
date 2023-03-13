# Leauge_of_Icons
---
A classifying deep learning model that distinguishes between league of legends portraits from other video game portraits
Agile Project: League of legends icon identifier
Project vision –
To make a league of legends icon identifier, using pytorch machine learning. The goal is to create a model that correctly identifies actual league of legends icons from a collection of other video game portraits/ abilities. 
How long should it take? – 
10 hours?

# Project roadmap
## 01/03 – Goal: Set the foundations.
Create conda environment.
set up project on git along with an initialised folder.
retrieve images for classifying.
Metrics – have over 100 league images and 100 non-league images, have a conda environment and initialised git folder
### Progress – 
building a webscraper but the website I wanted to use has lazy loading images meaning only 3 maximum images get pulled, tried first with Scrapy and BS4, now trying selenium as that loads the page like a user. It worked, images have been acrued

## 06/03 – Goal:  Classifying data.
Organise data into non-league and league images.
Determine test, dev and train sizes and justify why you are using those splits.
Create python pages for each ‘class’ of function – loading, processing, model, etc. as well as test pages 
Integrated testing within git push
Metrics - Finished these tasks – git commit
### Progress – 
did not create a page for each function since I was not sure as to what the functions would be. 
Doubled league data set by flipping images horizontally.

## 09/03 & 10/03 – Goal: Build you model.
Learn how to use pytorch to build your model
Start with a 2 layer NN, using ReLU for hidden and Sigmoid for L
Make layer an adjustable value for optimisation.
Metrics – build a NN that functions with an accuracy of at least 70% - git commit.
### Progress – 
sorted data into test train split, able to access data and done a bit of reading on how pytorch works. The split as described was 75/12.5/12.5 as the data set is not massive but I still want to dedicate as much data as possible to training.
Had an issue with the NN not recognising the sizes of the images, but then I removed the alpha channel from the non-league images.
Done! Produced a model with ~95% accuracy. However, I believe I could make it less computationally draining as well as correcting the League images to remove an easy bias (the black border). 


## 13/03 – Goal: Modify the data and determine if optimisation and regularisation helps.
Try applying He initialisation for the parameter weights,
Apply DO regularisation to ‘compress’ large amount  of inputs from images.   
Remove black border from league images.
Edit READ.ME and post on github
### Progress - 
deleted the borders and accuracy dropped instantly! 64%
Messed around with hyperparameters and since bias was high but variance was okay, I trained the model for longer, resulting in an 83% accurate model. 
He initialisation is already applied by default to layers, so that is unnecessary. 
Tweaked hyperparameters more, added another layer to the network and ran it for much longer. Now 83% train and 81% dev! Low variance but still increased bias, don’t think I can increase without more data.
Decreased Bias by including DO regularisation. 

#
# Code
## Pre-requisites

- Create an image folder, with distinct names seperating labels from one another - League = 1, Other = 0. Naming specifity can be adjusted in CSV creation

-  Make sure you have set up a conda environment and downloaded the required packages.

- In neural_net.py adjust the hyperparameters as required

## Running the code
1. Run the CSV creation script, to create a CSV in which file names are seperated and indexed against their values.

2. Use the train_and_save script to create your model and save it to your system. 

3. With load_and_test_model you can then load that model and test it against either your development or test data set.

4. Enjoy!

