Project Brief

Background:

An Online Jewelry Brand would like to add a feature where users can upload a picture of their jewelry and the system will recommend a similar product

Objectives:

Create a recommendation engine able to identify, classify and output similar mejuri products to that of the input images

This was accomplished by hooking into a specific layer of a convolutional neural network and comparing features of images to determine the best recommendations

DataBunch Compiled of Mejuri Site Product Images:

Train Data consisted of 576 images with a validation set of 212 images
19 classes, based off each unique collection of jewelry 

Parameter & Transformation Adjustments:

Mostly default parameters
Epochs: 7
Learning rate range  of 7 e-4 to 7 e-3
Batch Size of 32 for Memory purposes

Using Resnet 50 Resulted in Accuracy of 70%:

Could be improved with Non-mejuri Jewelry Images

Resnet 50 is a model that is pre-trained on the ImageNet Database containing 1.5 million pictures and 30,000 different classifications

2nd Data Bunch Required to Compare Similarity:

384 stand-alone jewelry images with no background

Used to compare the input image features to this databunch

Pytorch Hook Function Allowed Collection of Outputs from First Fully Connected Layer of Neural Network:

Produces tensors representing patterns in each image

Accuracy-Speed Trade-Off, ANNOY vs. Cosine Similarity:

App Deployment Required 4.4 GB Docker Image to be Pushed to AWS:

Final Product:

Summary:

Currently Classify Images with 70% accuracy and recommend 5 most similar images to a given input

Was accomplished by taking the output of the fully connected layer in a convolutional neural net and comparing that to input images evaluated the same way

AWS costs $$$

Future Work:

Look into Different Methods of Calculating Image Similarities

Further Parameter Optimization Required, Bracelet & Ring Issue

Compatibility with Phones?


