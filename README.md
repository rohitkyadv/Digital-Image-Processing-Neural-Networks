# Digital-Image-Processing-Neural-Networks

### For Datasets please go to my google drive https://drive.google.com/drive/folders/13Ge-kJuz3VN0VJOwmYoop1X5rdriuF6R?usp=sharing


## Task Description
Our task is we must classify the objects in an image of the road for self-driving cars. This is not image classification, this is pixel-wise classification also known as image segmentation. Self-driving cars are not common yet because the AI is not good enough yet. This image segmentation problem is a large part of self driving car AI. This dataset has 5 sets of 1000 images and the corresponding segmentation images This dataset contains input images and labeled semantic segmentations captured via the CARLA self-driving car simulator. Dataset is available online on Kaggle at the link given below.

Data Source: https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge


## Method Description
For this problem I am going to use a deep learning approach. There are many solutions to image segmentation but we are going to use the Encoder-Decoder Network because it is currently the best performing approach and variations on this architecture can produce exciting results as evidenced by the many papers published about the design like U-net[1], V-net[2],  SegNet[3], and LinkNet[4].

## Deep Convolutional Encoder-Decoder Network
I will build a machine learning model similar to an Encoder-Decoder Network that takes an image as input and outputs an image that has categorized the input into regions. I will then train and test it on our dataset to identify semantic segmentation of cars, roads, etc. in the image. The encoder is going to take the input image and generate a high dimensional feature vector whereas the decoder will take a high-dimensional feature vector and generates a semantic segmentation mask. Unfortunately, when the encoder creates the high-dimensional feature vector, the spatial information is lost, so in order to recreate the image with the semantic segmentation mask, we will need to bring the outputs from earlier to the decoder so it can learn to combine the spatial information with the feature information. This is like if the encoders know “where” the objects are but they don’t know “what” the objects are, and the decoder knows “what” the objects are but doesn’t know “where” the objects are. In order to get our segmentation mask we must combine them so we know “where and what” the objects are. We will combine them using simple concatenation of the outputs so the next layer of the network can learn from both inputs simultaneously.

## Network Architecture
The first half of the model will be the encoder. In the encoder I will perform convolutions, normalization, activation, saving the outputs for later, and max pooling. The second half of the model will be the decoder. In the decoder I will perform upsampling, concatenation of earlier saved outputs, convolutions, normalization and activation. Between the encoder and decoder I will add some convolutional layers to learn from the high-dimensional feature vector. After the decoder I will add a convolutional layer with a sigmoid function and 13 output channels to perform classification for each pixel. The general idea is in the encoder, features will be encoded, and in the decoder, the location of the features will be decoded. 

## Experimental Results
After solving the problems of not enough RAM or VRAM, the model is running and takes about 30 minutes to train for 5 epochs on the dataset. Before implementing a validation set, the model’s outputs were so accurate that I was sure it was overfitting and memorizing the dataset because it perfectly outlined every object in the image. But after adding a validation set and evaluating the model on the testing set, it was proven that it could generalize and get accurate on all the data. Below are the graphs of metrics over the training time as wells as a table of the metrics of the trained network on the training, validation, and testing sets.  


## Conclusions
The Encoder-Decoder Network architecture is extremely powerful for image segmentation. The results of this project are surprisingly accurate and exciting. For bonus points, I searched for publications about this data set but only found one notebook on Kaggle which reported a binary crossentropy loss of 0.0348 on the validation set after 100 epochs (https://www.kaggle.com/wangmo/self-driving-cars-road-segmentation-task). My network only had time to train for 5 epochs and I achieved a categorical crossentropy loss of 0.09358 on the validation set. I think with more computing power and time I could add layers and epochs and achieve a lower loss. I also found a paper about image segmentation for self-driving cars[5] which used a different dataset and reported IoU for 7 cutting-edge networks, and the best was 0.653. My network has an IoU of 0.8294 which is much better. This is probably because my network was not trained on real images which would have been harder. Overall I think my network is comparable to cutting-edge networks because it is a faithful variation on the Encoder-Decoder architecture which produces amazing results.
