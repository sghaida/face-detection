
# Data Scientist Nanodegree

## Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 


This notebook walks you through one of the most popular Udacity projects across machine learning and artificial intellegence nanodegree programs.  The goal is to classify images of dogs according to their breed.  

If you are looking for a more guided capstone project related to deep learning and convolutional neural networks, this might be just it.  Notice that even if you follow the notebook to creating your classifier, you must still create a blog post or deploy an application to fulfill the requirements of the capstone project.

Also notice, you may be able to use only parts of this notebook (for example certain coding portions or the data) without completing all parts and still meet all requirements of the capstone project.

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
- `dog_names` - list of string-valued dog breed names for translating labels


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('../../../data/dog_images/train')
valid_files, valid_targets = load_dataset('../../../data/dog_images/valid')
test_files, test_targets = load_dataset('../../../data/dog_images/test')

# load list of dog names
dog_names = [item[20:-1].split('.')[-1] for item in sorted(glob("../../../data/dog_images/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    Using TensorFlow backend.


    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.


### Import Human Dataset

In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.


```python
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("../../../data/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
```

    There are 13233 total human images.


---
<a id='step1'></a>
## Step 1: Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](/images/output_5_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
human_count = 0
for img in human_files_short:
    if face_detector(img):
        human_count += 1
percentage = (human_count/len(human_files_short)) * 100
print('Percentage of humans classified as people: {}%'.format(percentage))

dog_count = 0
for img in dog_files_short:
    if face_detector(img):
        dog_count += 1
percentage = (dog_count/len(dog_files_short)) * 100
print('Percentage of dogs missclassified as people: {}%'.format(percentage))


```

    Percentage of humans classified as people: 100.0%
    Percentage of dogs missclassified as people: 11.0%


__Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

__Answer:__

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.


```python
## (Optional) TODO: Report the performance of another  
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.
```

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape 

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 


```python
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```


```python
ResNet50_predict_labels('images/sample_cnn.png')
```




    916



### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```


```python
dog_detector('images/American_water_spaniel_00648.jpg')
```




    True



### (IMPLEMENTATION) Assess the Dog Detector

__Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
human_count = 0    
for img in human_files_short:
    isDog = dog_detector(img)
    if isDog:
        human_count += 1
    percentage = (human_count/len(human_files_short)) * 100
print('Percentage of humans misclassified as dogs:: {}%'.format(percentage))

dog_count = 0
for img in dog_files_short:
    isDog = dog_detector(img)
    if isDog:
        dog_count += 1
    percentage = (dog_count/len(dog_files_short)) * 100
print('Percentage of dogs correctly classified as dogs: {}%'.format(percentage))
```

    Percentage of humans misclassified as dogs:: 0.0%
    Percentage of dogs correctly classified as dogs: 100.0%


---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train. 

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun! 

### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|██████████| 6680/6680 [01:24<00:00, 78.97it/s] 
    100%|██████████| 835/835 [00:09<00:00, 87.66it/s] 
    100%|██████████| 836/836 [00:09<00:00, 87.99it/s] 


### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)
           
__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

__Answer:__ 

### Image Augmentation


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

model = Sequential()

### TODO: Define your architecture.
model.add(BatchNormalization(input_shape=(224, 224, 3)))
model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Dense(133, activation='softmax'))

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    batch_normalization_1 (Batch (None, 224, 224, 3)       12        
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 222, 222, 16)      448       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 111, 111, 16)      0         
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 111, 111, 16)      64        
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 109, 109, 32)      4640      
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 54, 54, 32)        0         
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 54, 54, 32)        128       
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 52, 52, 64)        18496     
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 26, 26, 64)        0         
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 26, 26, 64)        256       
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 24, 24, 128)       73856     
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 12, 12, 128)       0         
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 12, 12, 128)       512       
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 10, 10, 256)       295168    
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 5, 5, 256)         0         
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 5, 5, 256)         1024      
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 133)               34181     
    =================================================================
    Total params: 428,785
    Trainable params: 427,787
    Non-trainable params: 998
    _________________________________________________________________


### Compile the Model


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
from keras.preprocessing.image import ImageDataGenerator

# create and configure augmented image generator
datagen = ImageDataGenerator(
    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
    horizontal_flip=True) # randomly flip images horizontally

# fit augmented image generator on data
datagen.fit(train_tensors)
```


```python
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 10
batch_size = 20

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

# model.fit(train_tensors, train_targets, 
#           validation_data=(valid_tensors, valid_targets),
#           epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
history = model.fit_generator(datagen.flow(train_tensors, train_targets, batch_size=batch_size),
                    validation_data=(valid_tensors, valid_targets), 
                    steps_per_epoch=train_tensors.shape[0] // batch_size,
                    epochs=epochs, callbacks=[checkpointer], verbose=1)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

    Epoch 1/10
    333/334 [============================>.] - ETA: 0s - loss: 4.5359 - acc: 0.0449Epoch 00001: val_loss improved from inf to 4.65084, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 56s 169ms/step - loss: 4.5356 - acc: 0.0452 - val_loss: 4.6508 - val_acc: 0.0359
    Epoch 2/10
    333/334 [============================>.] - ETA: 0s - loss: 4.1769 - acc: 0.0824Epoch 00002: val_loss improved from 4.65084 to 4.20599, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 55s 163ms/step - loss: 4.1766 - acc: 0.0822 - val_loss: 4.2060 - val_acc: 0.0766
    Epoch 3/10
    333/334 [============================>.] - ETA: 0s - loss: 3.9259 - acc: 0.1188Epoch 00003: val_loss improved from 4.20599 to 3.92569, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 55s 164ms/step - loss: 3.9257 - acc: 0.1187 - val_loss: 3.9257 - val_acc: 0.1269
    Epoch 4/10
    333/334 [============================>.] - ETA: 0s - loss: 3.6467 - acc: 0.1686Epoch 00004: val_loss improved from 3.92569 to 3.64634, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 54s 163ms/step - loss: 3.6467 - acc: 0.1686 - val_loss: 3.6463 - val_acc: 0.1749
    Epoch 5/10
    333/334 [============================>.] - ETA: 0s - loss: 3.4122 - acc: 0.2098Epoch 00005: val_loss did not improve
    334/334 [==============================] - 55s 163ms/step - loss: 3.4121 - acc: 0.2096 - val_loss: 3.7965 - val_acc: 0.1377
    Epoch 6/10
    333/334 [============================>.] - ETA: 0s - loss: 3.2098 - acc: 0.2405Epoch 00006: val_loss improved from 3.64634 to 3.42942, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 54s 163ms/step - loss: 3.2090 - acc: 0.2404 - val_loss: 3.4294 - val_acc: 0.2048
    Epoch 7/10
    333/334 [============================>.] - ETA: 0s - loss: 3.0039 - acc: 0.2872Epoch 00007: val_loss improved from 3.42942 to 3.17068, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 54s 161ms/step - loss: 3.0020 - acc: 0.2882 - val_loss: 3.1707 - val_acc: 0.2407
    Epoch 8/10
    333/334 [============================>.] - ETA: 0s - loss: 2.8290 - acc: 0.3305Epoch 00008: val_loss did not improve
    334/334 [==============================] - 54s 163ms/step - loss: 2.8286 - acc: 0.3304 - val_loss: 3.2811 - val_acc: 0.2228
    Epoch 9/10
    333/334 [============================>.] - ETA: 0s - loss: 2.6645 - acc: 0.3614Epoch 00009: val_loss improved from 3.17068 to 2.89398, saving model to saved_models/weights.best.from_scratch.hdf5
    334/334 [==============================] - 55s 164ms/step - loss: 2.6633 - acc: 0.3617 - val_loss: 2.8940 - val_acc: 0.2874
    Epoch 10/10
    333/334 [============================>.] - ETA: 0s - loss: 2.5059 - acc: 0.3938Epoch 00010: val_loss did not improve
    334/334 [==============================] - 54s 162ms/step - loss: 2.5055 - acc: 0.3937 - val_loss: 2.9748 - val_acc: 0.2659



![png](images/output_33_1.png)



![png](images/output_33_2.png)


### Load the Model with the Best Validation Loss


```python
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### Test the Model

Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.


```python
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 27.9904%


---
<a id='step4'></a>
## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.

### Obtain Bottleneck Features


```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_2 ( (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229
    Trainable params: 68,229
    Non-trainable params: 0
    _________________________________________________________________


### Compile the Model


```python
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### Train the Model


```python
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

history = VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6460/6680 [============================>.] - ETA: 0s - loss: 12.4662 - acc: 0.1144Epoch 00001: val_loss improved from inf to 11.14875, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 281us/step - loss: 12.4218 - acc: 0.1166 - val_loss: 11.1488 - val_acc: 0.2000
    Epoch 2/20
    6480/6680 [============================>.] - ETA: 0s - loss: 10.4276 - acc: 0.2619Epoch 00002: val_loss improved from 11.14875 to 10.27843, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s 221us/step - loss: 10.4397 - acc: 0.2617 - val_loss: 10.2784 - val_acc: 0.2683
    Epoch 3/20
    6440/6680 [===========================>..] - ETA: 0s - loss: 9.8627 - acc: 0.3224Epoch 00003: val_loss improved from 10.27843 to 9.98325, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s 221us/step - loss: 9.8456 - acc: 0.3237 - val_loss: 9.9832 - val_acc: 0.2970
    Epoch 4/20
    6460/6680 [============================>.] - ETA: 0s - loss: 9.5869 - acc: 0.3560Epoch 00004: val_loss improved from 9.98325 to 9.77055, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s 221us/step - loss: 9.5545 - acc: 0.3576 - val_loss: 9.7706 - val_acc: 0.3066
    Epoch 5/20
    6520/6680 [============================>.] - ETA: 0s - loss: 9.4035 - acc: 0.3839Epoch 00005: val_loss did not improve
    6680/6680 [==============================] - 1s 218us/step - loss: 9.3961 - acc: 0.3843 - val_loss: 9.8295 - val_acc: 0.3174
    Epoch 6/20
    6620/6680 [============================>.] - ETA: 0s - loss: 9.3395 - acc: 0.3952Epoch 00006: val_loss did not improve
    6680/6680 [==============================] - 2s 249us/step - loss: 9.3250 - acc: 0.3961 - val_loss: 9.7860 - val_acc: 0.3293
    Epoch 7/20
    6500/6680 [============================>.] - ETA: 0s - loss: 9.2926 - acc: 0.4078Epoch 00007: val_loss did not improve
    6680/6680 [==============================] - 2s 247us/step - loss: 9.2839 - acc: 0.4081 - val_loss: 9.8139 - val_acc: 0.3269
    Epoch 8/20
    6640/6680 [============================>.] - ETA: 0s - loss: 9.2622 - acc: 0.4120Epoch 00008: val_loss did not improve
    6680/6680 [==============================] - 2s 249us/step - loss: 9.2599 - acc: 0.4123 - val_loss: 9.7732 - val_acc: 0.3293
    Epoch 9/20
    6480/6680 [============================>.] - ETA: 0s - loss: 9.2031 - acc: 0.4188Epoch 00009: val_loss improved from 9.77055 to 9.59475, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 249us/step - loss: 9.2099 - acc: 0.4183 - val_loss: 9.5948 - val_acc: 0.3437
    Epoch 10/20
    6600/6680 [============================>.] - ETA: 0s - loss: 9.0870 - acc: 0.4186Epoch 00010: val_loss improved from 9.59475 to 9.50656, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 251us/step - loss: 9.0699 - acc: 0.4193 - val_loss: 9.5066 - val_acc: 0.3401
    Epoch 11/20
    6500/6680 [============================>.] - ETA: 0s - loss: 8.8234 - acc: 0.4369Epoch 00011: val_loss improved from 9.50656 to 9.38254, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 283us/step - loss: 8.8435 - acc: 0.4356 - val_loss: 9.3825 - val_acc: 0.3485
    Epoch 12/20
    6580/6680 [============================>.] - ETA: 0s - loss: 8.7648 - acc: 0.4448Epoch 00012: val_loss improved from 9.38254 to 9.22237, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 284us/step - loss: 8.7518 - acc: 0.4454 - val_loss: 9.2224 - val_acc: 0.3509
    Epoch 13/20
    6640/6680 [============================>.] - ETA: 0s - loss: 8.6016 - acc: 0.4571Epoch 00013: val_loss improved from 9.22237 to 9.19192, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 283us/step - loss: 8.6080 - acc: 0.4567 - val_loss: 9.1919 - val_acc: 0.3653
    Epoch 14/20
    6560/6680 [============================>.] - ETA: 0s - loss: 8.5934 - acc: 0.4620Epoch 00014: val_loss did not improve
    6680/6680 [==============================] - 2s 283us/step - loss: 8.5861 - acc: 0.4626 - val_loss: 9.1992 - val_acc: 0.3581
    Epoch 15/20
    6580/6680 [============================>.] - ETA: 0s - loss: 8.6150 - acc: 0.4619Epoch 00015: val_loss improved from 9.19192 to 9.10161, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 283us/step - loss: 8.5781 - acc: 0.4641 - val_loss: 9.1016 - val_acc: 0.3689
    Epoch 16/20
    6620/6680 [============================>.] - ETA: 0s - loss: 8.5585 - acc: 0.4657Epoch 00016: val_loss did not improve
    6680/6680 [==============================] - 2s 264us/step - loss: 8.5709 - acc: 0.4650 - val_loss: 9.2357 - val_acc: 0.3689
    Epoch 17/20
    6560/6680 [============================>.] - ETA: 0s - loss: 8.5603 - acc: 0.4660Epoch 00017: val_loss did not improve
    6680/6680 [==============================] - 2s 250us/step - loss: 8.5652 - acc: 0.4657 - val_loss: 9.2219 - val_acc: 0.3629
    Epoch 18/20
    6520/6680 [============================>.] - ETA: 0s - loss: 8.5250 - acc: 0.4644Epoch 00018: val_loss improved from 9.10161 to 9.08865, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 254us/step - loss: 8.5076 - acc: 0.4653 - val_loss: 9.0887 - val_acc: 0.3665
    Epoch 19/20
    6600/6680 [============================>.] - ETA: 0s - loss: 8.3865 - acc: 0.4703Epoch 00019: val_loss improved from 9.08865 to 9.02123, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 256us/step - loss: 8.3852 - acc: 0.4704 - val_loss: 9.0212 - val_acc: 0.3749
    Epoch 20/20
    6640/6680 [============================>.] - ETA: 0s - loss: 8.3397 - acc: 0.4747Epoch 00020: val_loss improved from 9.02123 to 8.99127, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 249us/step - loss: 8.3506 - acc: 0.4740 - val_loss: 8.9913 - val_acc: 0.3737



```python
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```


![png](images/output_46_0.png)



![png](images/output_46_1.png)


### Load the Model with the Best Validation Loss


```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.


```python
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)

print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 38.7560%


### Predict Dog Breed with the Model


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```


```python
VGG16_predict_breed('images/Curly-coated_retriever_03896.jpg')
```




    'Curly-coated_retriever'



---
<a id='step5'></a>
## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz
    
where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the `bottleneck_features/` folder in the repository.

### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
import numpy as np

### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_ResNet50 = bottleneck_features['train']
valid_ResNet50 = bottleneck_features['valid']
test_ResNet50 = bottleneck_features['test']
```

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        <your model's name>.summary()
   
__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ 

I decided to follow a similar approach as the previous feature extraction example using the VGG16 network. This time I decided to experiment with the Resnet50 network. I passed the bottleneck features into a GlobalAveragePooling2D layer, which was then followed by a Dropout layer with a fairly large rate. This did a decent job helping with the overfitting issue. The data was then passed into a Dense layer of 133 nodes, 1 for each breed, with a softmax activation. i used Adamax with learning rate of 0.002 which allows slow but steady learning.

While a test accuracy of 84% is decent, we can see from the graphs below, the network is clearly overfitting. Probably the most effective way to solve this issue given a small training set of images is data augmentation.

Given the current structure of the project and how the features are being extracted:

bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')

I realized I would not be able to use data augmentation with this setup and I would have to augment my data and pass it through the entire network (computationally expensive).



```python
### TODO: Define your architecture.
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

ResNet_model = Sequential()
ResNet_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
ResNet_model.add(Dropout(0.3))
ResNet_model.add(Dense(133, activation='softmax'))

ResNet_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_3 ( (None, 2048)              0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 2048)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 133)               272517    
    =================================================================
    Total params: 272,517
    Trainable params: 272,517
    Non-trainable params: 0
    _________________________________________________________________


### (IMPLEMENTATION) Compile the Model


```python
### TODO: Compile the model.
from keras.optimizers import Adam, Adamax
ResNet_model.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=0.002), metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
### TODO: Train the model.
from keras.callbacks import ModelCheckpoint  

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best_adamax.ResNet50.hdf5', 
                               verbose=1, save_best_only=True)

epochs = 30
batch_size = 64

history = ResNet_model.fit(train_ResNet50, train_targets, 
          validation_data=(valid_ResNet50, valid_targets),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/30
    6336/6680 [===========================>..] - ETA: 0s - loss: 2.8828 - acc: 0.3621Epoch 00001: val_loss improved from inf to 1.39232, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 154us/step - loss: 2.8237 - acc: 0.3708 - val_loss: 1.3923 - val_acc: 0.6563
    Epoch 2/30
    6144/6680 [==========================>...] - ETA: 0s - loss: 1.0957 - acc: 0.7316Epoch 00002: val_loss improved from 1.39232 to 0.99225, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 86us/step - loss: 1.0780 - acc: 0.7356 - val_loss: 0.9922 - val_acc: 0.7521
    Epoch 3/30
    6400/6680 [===========================>..] - ETA: 0s - loss: 0.7471 - acc: 0.8155Epoch 00003: val_loss improved from 0.99225 to 0.84519, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 86us/step - loss: 0.7426 - acc: 0.8181 - val_loss: 0.8452 - val_acc: 0.7749
    Epoch 4/30
    6080/6680 [==========================>...] - ETA: 0s - loss: 0.5807 - acc: 0.8645Epoch 00004: val_loss improved from 0.84519 to 0.75760, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 87us/step - loss: 0.5771 - acc: 0.8644 - val_loss: 0.7576 - val_acc: 0.7904
    Epoch 5/30
    6656/6680 [============================>.] - ETA: 0s - loss: 0.4630 - acc: 0.8951Epoch 00005: val_loss improved from 0.75760 to 0.70089, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 89us/step - loss: 0.4633 - acc: 0.8948 - val_loss: 0.7009 - val_acc: 0.8000
    Epoch 6/30
    6144/6680 [==========================>...] - ETA: 0s - loss: 0.3811 - acc: 0.9189Epoch 00006: val_loss improved from 0.70089 to 0.67061, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 88us/step - loss: 0.3845 - acc: 0.9175 - val_loss: 0.6706 - val_acc: 0.8192
    Epoch 7/30
    6144/6680 [==========================>...] - ETA: 0s - loss: 0.3343 - acc: 0.9248Epoch 00007: val_loss improved from 0.67061 to 0.63895, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 86us/step - loss: 0.3366 - acc: 0.9240 - val_loss: 0.6389 - val_acc: 0.8275
    Epoch 8/30
    6336/6680 [===========================>..] - ETA: 0s - loss: 0.2838 - acc: 0.9427Epoch 00008: val_loss improved from 0.63895 to 0.61724, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 86us/step - loss: 0.2843 - acc: 0.9422 - val_loss: 0.6172 - val_acc: 0.8204
    Epoch 9/30
    6144/6680 [==========================>...] - ETA: 0s - loss: 0.2519 - acc: 0.9536Epoch 00009: val_loss improved from 0.61724 to 0.60353, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 88us/step - loss: 0.2501 - acc: 0.9536 - val_loss: 0.6035 - val_acc: 0.8323
    Epoch 10/30
    6400/6680 [===========================>..] - ETA: 0s - loss: 0.2205 - acc: 0.9586Epoch 00010: val_loss improved from 0.60353 to 0.59302, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 85us/step - loss: 0.2197 - acc: 0.9588 - val_loss: 0.5930 - val_acc: 0.8287
    Epoch 11/30
    6592/6680 [============================>.] - ETA: 0s - loss: 0.1921 - acc: 0.9669Epoch 00011: val_loss improved from 0.59302 to 0.58371, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 90us/step - loss: 0.1919 - acc: 0.9671 - val_loss: 0.5837 - val_acc: 0.8311
    Epoch 12/30
    6144/6680 [==========================>...] - ETA: 0s - loss: 0.1688 - acc: 0.9740Epoch 00012: val_loss improved from 0.58371 to 0.58056, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 88us/step - loss: 0.1685 - acc: 0.9744 - val_loss: 0.5806 - val_acc: 0.8287
    Epoch 13/30
    6464/6680 [============================>.] - ETA: 0s - loss: 0.1519 - acc: 0.9785Epoch 00013: val_loss improved from 0.58056 to 0.56359, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 89us/step - loss: 0.1531 - acc: 0.9780 - val_loss: 0.5636 - val_acc: 0.8311
    Epoch 14/30
    6656/6680 [============================>.] - ETA: 0s - loss: 0.1385 - acc: 0.9815Epoch 00014: val_loss improved from 0.56359 to 0.56179, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 88us/step - loss: 0.1386 - acc: 0.9814 - val_loss: 0.5618 - val_acc: 0.8335
    Epoch 15/30
    6656/6680 [============================>.] - ETA: 0s - loss: 0.1238 - acc: 0.9851Epoch 00015: val_loss improved from 0.56179 to 0.54609, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 87us/step - loss: 0.1240 - acc: 0.9849 - val_loss: 0.5461 - val_acc: 0.8383
    Epoch 16/30
    6400/6680 [===========================>..] - ETA: 0s - loss: 0.1147 - acc: 0.9872Epoch 00016: val_loss improved from 0.54609 to 0.54013, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 90us/step - loss: 0.1151 - acc: 0.9868 - val_loss: 0.5401 - val_acc: 0.8479
    Epoch 17/30
    6592/6680 [============================>.] - ETA: 0s - loss: 0.1060 - acc: 0.9873Epoch 00017: val_loss improved from 0.54013 to 0.53680, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 89us/step - loss: 0.1060 - acc: 0.9873 - val_loss: 0.5368 - val_acc: 0.8443
    Epoch 18/30
    6336/6680 [===========================>..] - ETA: 0s - loss: 0.0929 - acc: 0.9896Epoch 00018: val_loss did not improve
    6680/6680 [==============================] - 1s 83us/step - loss: 0.0940 - acc: 0.9892 - val_loss: 0.5372 - val_acc: 0.8479
    Epoch 19/30
    6208/6680 [==========================>...] - ETA: 0s - loss: 0.0864 - acc: 0.9910Epoch 00019: val_loss improved from 0.53680 to 0.53417, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 86us/step - loss: 0.0868 - acc: 0.9907 - val_loss: 0.5342 - val_acc: 0.8407
    Epoch 20/30
    6144/6680 [==========================>...] - ETA: 0s - loss: 0.0788 - acc: 0.9933Epoch 00020: val_loss improved from 0.53417 to 0.52865, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 87us/step - loss: 0.0790 - acc: 0.9936 - val_loss: 0.5286 - val_acc: 0.8467
    Epoch 21/30
    6272/6680 [===========================>..] - ETA: 0s - loss: 0.0754 - acc: 0.9930Epoch 00021: val_loss did not improve
    6680/6680 [==============================] - 1s 84us/step - loss: 0.0754 - acc: 0.9930 - val_loss: 0.5310 - val_acc: 0.8431
    Epoch 22/30
    6208/6680 [==========================>...] - ETA: 0s - loss: 0.0705 - acc: 0.9931Epoch 00022: val_loss did not improve
    6680/6680 [==============================] - 1s 83us/step - loss: 0.0704 - acc: 0.9931 - val_loss: 0.5377 - val_acc: 0.8359
    Epoch 23/30
    6336/6680 [===========================>..] - ETA: 0s - loss: 0.0638 - acc: 0.9959Epoch 00023: val_loss did not improve
    6680/6680 [==============================] - 1s 83us/step - loss: 0.0637 - acc: 0.9958 - val_loss: 0.5366 - val_acc: 0.8479
    Epoch 24/30
    6144/6680 [==========================>...] - ETA: 0s - loss: 0.0590 - acc: 0.9966Epoch 00024: val_loss did not improve
    6680/6680 [==============================] - 1s 84us/step - loss: 0.0599 - acc: 0.9961 - val_loss: 0.5421 - val_acc: 0.8347
    Epoch 25/30
    6272/6680 [===========================>..] - ETA: 0s - loss: 0.0566 - acc: 0.9951Epoch 00025: val_loss improved from 0.52865 to 0.52568, saving model to saved_models/weights.best_adamax.ResNet50.hdf5
    6680/6680 [==============================] - 1s 86us/step - loss: 0.0576 - acc: 0.9948 - val_loss: 0.5257 - val_acc: 0.8395
    Epoch 26/30
    6208/6680 [==========================>...] - ETA: 0s - loss: 0.0530 - acc: 0.9961Epoch 00026: val_loss did not improve
    6680/6680 [==============================] - 1s 84us/step - loss: 0.0528 - acc: 0.9963 - val_loss: 0.5319 - val_acc: 0.8467
    Epoch 27/30
    6528/6680 [============================>.] - ETA: 0s - loss: 0.0480 - acc: 0.9972Epoch 00027: val_loss did not improve
    6680/6680 [==============================] - 1s 90us/step - loss: 0.0480 - acc: 0.9972 - val_loss: 0.5274 - val_acc: 0.8443
    Epoch 28/30
    6208/6680 [==========================>...] - ETA: 0s - loss: 0.0464 - acc: 0.9968Epoch 00028: val_loss did not improve
    6680/6680 [==============================] - 1s 94us/step - loss: 0.0465 - acc: 0.9966 - val_loss: 0.5261 - val_acc: 0.8515
    Epoch 29/30
    6656/6680 [============================>.] - ETA: 0s - loss: 0.0425 - acc: 0.9974Epoch 00029: val_loss did not improve
    6680/6680 [==============================] - 1s 94us/step - loss: 0.0425 - acc: 0.9975 - val_loss: 0.5289 - val_acc: 0.8527
    Epoch 30/30
    6208/6680 [==========================>...] - ETA: 0s - loss: 0.0402 - acc: 0.9977Epoch 00030: val_loss did not improve
    6680/6680 [==============================] - 1s 94us/step - loss: 0.0403 - acc: 0.9979 - val_loss: 0.5382 - val_acc: 0.8503



```python
opt = Adamax(lr=0.0002)
epochs = 5
batch_size = 64

history = ResNet_model.fit(train_ResNet50, train_targets, 
          validation_data=(valid_ResNet50, valid_targets),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/5
    6656/6680 [============================>.] - ETA: 0s - loss: 0.0374 - acc: 0.9964Epoch 00001: val_loss did not improve
    6680/6680 [==============================] - 1s 139us/step - loss: 0.0373 - acc: 0.9964 - val_loss: 0.5313 - val_acc: 0.8431
    Epoch 2/5
    6144/6680 [==========================>...] - ETA: 0s - loss: 0.0355 - acc: 0.9982Epoch 00002: val_loss did not improve
    6680/6680 [==============================] - 1s 94us/step - loss: 0.0357 - acc: 0.9982 - val_loss: 0.5348 - val_acc: 0.8419
    Epoch 3/5
    6272/6680 [===========================>..] - ETA: 0s - loss: 0.0331 - acc: 0.9976Epoch 00003: val_loss did not improve
    6680/6680 [==============================] - 1s 93us/step - loss: 0.0332 - acc: 0.9978 - val_loss: 0.5327 - val_acc: 0.8431
    Epoch 4/5
    6464/6680 [============================>.] - ETA: 0s - loss: 0.0343 - acc: 0.9975Epoch 00004: val_loss did not improve
    6680/6680 [==============================] - 1s 94us/step - loss: 0.0341 - acc: 0.9976 - val_loss: 0.5341 - val_acc: 0.8431
    Epoch 5/5
    6272/6680 [===========================>..] - ETA: 0s - loss: 0.0319 - acc: 0.9970Epoch 00005: val_loss did not improve
    6680/6680 [==============================] - 1s 94us/step - loss: 0.0319 - acc: 0.9972 - val_loss: 0.5308 - val_acc: 0.8479



```python
# Plotting
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```


![png](images/output_63_0.png)



![png](images/output_63_1.png)


### (IMPLEMENTATION) Load the Model with the Best Validation Loss


```python
### TODO: Load the model weights with the best validation loss.
ResNet_model.load_weights('saved_models/weights.best_adamax.ResNet50.hdf5')
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.


```python
### TODO: Calculate classification accuracy on the test dataset.

# get index of predicted dog breed for each image in test set
ResNet50_predictions = [np.argmax(ResNet_model.predict(np.expand_dims(feature, axis=0))) for feature in test_ResNet50]

# report test accuracy
test_accuracy = 100*np.sum(np.array(ResNet50_predictions)==np.argmax(test_targets, axis=1))/len(ResNet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 83.4928%


### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.  

Similar to the analogous function in Step 5, your function should have three steps:
1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

    extract_{network}
    
where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.


```python
def extract_Resnet50(tensor):
    from keras.applications.resnet50 import ResNet50, preprocess_input
    return ResNet50(weights='imagenet',pooling='avg', include_top=False).predict(preprocess_input(tensor))
```


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

from extract_bottleneck_features import *

def ResNet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = ResNet_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    breed = dog_names[np.argmax(predicted_vector)]
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgplot = plt.imshow(cv_rgb)
    if dog_detector(img_path) == True:
        return print("The breed of dog is a {}".format(breed))
    else:
        return print("If this person were a dog, the breed would be a {}".format(breed))
```


```python
ResNet50_predict_breed('images/American_water_spaniel_00648.jpg')
```

    The breed of dog is a Boykin_spaniel



![png](images/output_71_1.png)


---
<a id='step6'></a>
## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  

A sample image and output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_2.png)

This photo looks like an Afghan Hound.
### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def predict_breed(img_path):
    isDog = dog_detector(img_path)
    isPerson = face_detector(img_path)
    if isDog:
        print("Detected a dog")
        breed = ResNet50_predict_breed(img_path)
        return breed
    if isPerson:
        print("Detected a human face")
        breed = ResNet50_predict_breed(img_path)
        return breed
    else:
        print("No human face or dog detected")
        img = cv2.imread(img_path)
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgplot = plt.imshow(cv_rgb)
```

---
<a id='step7'></a>
## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ 


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

predict_breed('images/Curly-coated_retriever_03896.jpg')
```

    Detected a dog
    The breed of dog is a Curly-coated_retriever



![png](images/output_75_1.png)



```python
predict_breed('images/Brittany_02625.jpg')
```

    Detected a dog
    The breed of dog is a Brittany



![png](images/output_76_1.png)



```python
predict_breed('images/Labrador_retriever_06449.jpg')
```

    Detected a dog
    The breed of dog is a Labrador_retriever



![png](images/output_77_1.png)



```python
predict_breed('images/sample_human_2.png')
```

    Detected a human face
    If this person were a dog, the breed would be a Afghan_hound



![png](images/output_78_1.png)



```python
predict_breed('images/Welsh_springer_spaniel_08203.jpg')
```

    Detected a dog
    The breed of dog is a Welsh_springer_spaniel



![png](images/output_79_1.png)



```python
predict_breed('images/sample_human_output.png')
```

    Detected a human face
    If this person were a dog, the breed would be a English_toy_spaniel



![png](images/output_80_1.png)



```python

```
