---
output:
  pdf_document: default
  html_document: default
---
# Siamese Deep NN for object recognition in fashion industry

## Capstone Project Proposal
Pu He
Jan 7th, 2018

### Domain Background
I have always been interested in fashion industry and this project aims to train a deep siamese network that return the most similar product given a picture of the product. This is important when trying to merging data from different sources, different e-commerce website that sells fashion products. Also to sites like Amazon or Taobao, on one hand this functionality is important to themselves in order to keep track of what is being sold on their webiste. On the other hand, they already have the features released so that customers could just upload a picture of what they want to buy and found out whether it is on Amazon or not. This could also be a convenient way for generating sales since sometimes customers do not know the exact name of the products they are looking for but have a photo instead. 

### Problem Statement
There are millions of products being sold on sites like Amazon and there are plenty on online fashion store too. Due to limitation of data and computing power, I decide to focus on handbag category on one of the biggest online fashion store. The problem would reduce to train a deep learning model that returns the name of the handbag given a picture of the handbag. We could measure the error rate of the classification as performance measure.  

### Datasets and Inputs
The dataset I used is the colored picture I crawlled using scrapy package from a major fashion online store's webiste (prefer not to say the exact website name) and the corresponding unique product code associated with the handbag in the picture. 

The input images are in jpg format. I randomly shuffle them into 3 folders, train (70%), valid (15%) and test(15%). Similar to the deep learning project in this nanodegree, I organize the pictures so that pictures of the same product are under one folder with the folder name being thier product code. For exmaple, all training pictures of product '103245' are stored under folder './train/103245/'. 

This dataset is appropriate for our problem since the pictures are directly taken from a online fashion store. One concern might be that what user uploads can be quite different with what pictures online store presents in terms of lighting, angle, clairty and resolution etc.. To relieve a bit that concern we might augment our dataset a bit by adjusting light, doing random cropping and rotating etc.. We will describe them in detail in the final report. 


### Solution Statement
Given the number of pictures per category (around 5 pictures) and a relatively large number of categories, I think this problem is very suitable to apply one short learning techinque where for each category there are very few data points. Inspired by the one shot learning literature like [2] I decide to use Siamese deep learning architecture with pair loss or triplet loss as loss function to learn a similarity score for any pair of pcitures so that pictures of the same product would have higher similarity score while ones of differet products would have lower similarity scores. VGG deep face paper [1] uses triplet loss function while uses Siamese network structure to do one shot learning. To constrat with image classification model, we call this model here siamese object recognition model.

One alternative would be using CNN architecture to do standard image-classification. I don't think this is useful since there are too few points per category. And also one big draw back is that if we have one more product being sold on the website, we need to train the entire network once again, which can be very time-consuming. 

### Benchmark Model

I plan to use a shallow but fully connected neuron network model performing image-classification as a benchmark to my siamese object recognition model. I expect to see some performance improvement using siamese object recognition. 

### Evaluation Metrics
I decide to use two measures: accuracy and top-5 accuracy. For benchmark model which returns probability of being every single product for each test picture, we could rank from highest probability to the lowest. The accuracy of always choosing the highest probability one is defined as the accuracy and that is the standard accuracy used in classification. For top-5 accuracy, for each test image we check whether the true product category is within the top 5 categories with the highest probability. This is an easier task and I think this measure is also relevant since after users submit one picture, we want to return a list ranking from the most similar to least similar and as long as what the user wants are within the top-n, say top 5 then we should be fine. 

For our siamese object recognition model, we can also do the same thing. The siamese network would translate the pixel values of input picture into a vector and we will compute the similarity scores used in the training (say euclidean distance) between the test image and every single image in the training + validation data. We take the top 5 most similar pictures and use their categories as the prediction. If the true product category is among the cateogries of the most similar 5 pictures returned, then we mark it as correct. That will give us top-5 accuracy. If we always pick the most similar picture's category as our prediction, this will give us standard accuracy.

Of course we evalute those two measures on validation set to tune hyperparameters and finally after all things are done we evalute them on the test set. 

### Project Design

I decide to use Keras as the programming framework training on GTX 1080 Ti on my window desktop. 

1) proprocess input pictures, standardize resolutions and possibly do some data augmentation to generate more training data
2) Construct pairs for training the siamese object recognition model. 
I plan to include all pairs (p1, p2) where p1 and p2 are of the same product. So those training data will be ((p1,p2),1) where 1 denotes they are the same product. I also need negatve pairs and inspired by the VGG deep face paper for every single picture in the training set p1, we randomly sample one picture p2 that is not the same product and denote this training data point as ((p1,p2), 0) where 0 denotes that p1 and p3 are not the same product. In our data set this would generate relatively the same number of positive pairs ((p1,p2),1) as negative pairs ((p1,p2),0). Going through all ((p1,p2),y) is considered as one epoch.  
3) Pick several well known deep CNN network structure like inceptionV3, freeze some earlier layers' weights since the amount of data we have is too few to train a really deep network. Denote the entire trained CNN network as a function f that transforms pixal values in the first layer to a vector of real values output by the final layer. We take the abosolute difference between f(p1) and f(p2), mulitply it by a learned weight vector w and apply softmax to it to get a probability of whether p1 and p2 are pictures of the same product, i.e. 

$$
Probability(p_1,p_2 \textrm{ are the same product}) = softmax(w|f(p1) - f(p2)|)
$$ 

We consider the probability also as a similarity score between p1 and p2. 
4) For each training pair ((p1,p2), y) we calculate the cross entropy loss between the probability and the true label y and minimize this loss function over minibatches. We plan to do early stopping based on accuracy measure on validation set. Accuracy is as defined in Evaluation Metrics. 
5) Going back to 3), modify and network structure and fine tuning other hyperparameter to try to get better validation accuracy. 
6) After validation accuracy is good, evaluate accuracy and top-5 accuracy on test set. Accuracy and top-5 accuracy are as defined in Evaluation Metrics. 

### Data link
I provide a link to the data in my dropbox folder as below
https://www.dropbox.com/sh/kn2czhnwz0jj93i/AAAimCuNAKZIzxo_P5w5UwkNa?dl=0


### Reference
[1] Deep Face Recognition. Omkar M. Parkhi, Andrea Vedaldi and Andrew Zisserman \\
[2] Siamese Neural Networks for One-shot Image Recognition. Gregory Koch, Richard Zemel and Ruslan Salakhutdinov 
