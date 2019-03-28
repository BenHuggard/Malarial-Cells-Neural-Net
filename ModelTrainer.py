#this script trains a regression model to detect the presence of Plasmodium parasites
#in blood smear images of cells
#with minor tweaks this script could easily be reconfigured to train and 
#optimize on any image dataset

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import time
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split as tts

DIRECTORY = os.path.dirname(os.path.abspath(__file__))

DUMMY_INFECTED_PATH = DIRECTORY + "/dummy_data/Parasitized"
DUMMY_HEALTHY_PATH = DIRECTORY + "/dummy_data/Uninfected"

#very small dummy dataset for error checking
INFECTED_PATH = DIRECTORY + "/cell_images/Parasitized"
HEALTHY_PATH = DIRECTORY + "/cell_images/Uninfected"

#this will store the dimensions of the smallest image in the set
IMG_SIZE = [256, 256] #temporary placeholder data

def load_data(HEALTHY_PATH, INFECTED_PATH):
    #we begin by loading each image as a numpy array into a python list
    #after images have been resized we can convert this list into a np array
    
    IMG_COUNT = [0, 0]
    
    VALID_EXT = ".png"
    
    print("Loading data...")
    
    start_time = time.time()
    infected_imgs = []
    for file in os.listdir(INFECTED_PATH):
        file_ext = os.path.splitext(file)[1] #obtain extension on file
        if file_ext == VALID_EXT:
            image = Image.open(os.path.join(INFECTED_PATH, file))
            image_copy = np.array(image.copy())
            infected_imgs.append(image_copy)   
            image.close() #must use a copy and close so we do not get "Too many files open" error
    
    elapsed = round(time.time() - start_time)
    
    if __name__ == '__main__':
        print("-- Infected Images loaded in {} seconds --".format(elapsed))
    
    start_time = time.time()
    healthy_imgs = []
    for file in os.listdir(HEALTHY_PATH):
        file_ext = os.path.splitext(file)[1]
        if file_ext == VALID_EXT:
            image = Image.open(os.path.join(HEALTHY_PATH, file))
            image_copy = np.array(image.copy())
            healthy_imgs.append(image_copy)   
            image.close()

    elapsed = round(time.time() - start_time)
    if __name__ == '__main__':
        print("-- Healthy Images loaded in {} seconds --".format(elapsed))
    
    IMG_COUNT[0] = len(infected_imgs)
    IMG_COUNT[1] = len(healthy_imgs)
    
    #returns image sets as python lists of numpy arrays
    return healthy_imgs, infected_imgs, IMG_COUNT
            

#visualize data sizes
def visualize_data(healthy_imgs, infected_imgs):
    print("Healthy Cell: ")
    plt.imshow(healthy_imgs[4])
    plt.show()
    print("Infected Cell: ")
    plt.imshow(infected_imgs[100])
    plt.show()
    #right away we can see a difference in cell appearance
    
    array_sizes = [len(healthy_imgs), len(infected_imgs)]
    
    plt.bar(np.arange(len(array_sizes)), array_sizes)
    plt.xlabel("Cell status")
    plt.ylabel("Size of status data")
    plt.xticks(np.arange(len(array_sizes)), ["Healthy", "Infected"])
    
    print("# of Healthy cells: ", array_sizes[0])
    print("# of Infected cells: ", array_sizes[1])
    plt.show()
    #we have an equal number of infected and healthy images

def process_image(image, IMG_SIZE):
    #process an individual image as a np array
    
    temp = rgb2gray(image)
    temp = normalize(temp)
    temp = resize(temp, (IMG_SIZE[0], IMG_SIZE[1]))
    #conv2D layer takes in a 4D tensor, so we must add a dimension to our data
    return temp.reshape(IMG_SIZE[0], IMG_SIZE[1], 1)

def process_data(healthy_imgs, infected_imgs, IMG_COUNT, passed_size):
    #process, label sets of images
    
    combined_imgs = infected_imgs + healthy_imgs
    
    global IMG_SIZE
    
    start_time = time.time()
    print("")
    print("Processing data...")

    #first must find minimum values for image width, height for resizing
    
    if __name__ == '__main__':
        for image in combined_imgs:
            if len(image) < IMG_SIZE[0]:
                IMG_SIZE[0] = len(image)
            if len(image[0]) < IMG_SIZE[1]:
                IMG_SIZE[1] = len(image[0])
    else:
        IMG_SIZE[0] = passed_size[0]
        IMG_SIZE[1] = passed_size[1]
            
    if __name__ == '__main__':
        print("Dimensions to resize to: {}, {}".format(IMG_SIZE[0], IMG_SIZE[1]))
    
    #normalize/grayscale and resize
    for i in range(len(combined_imgs)):
        combined_imgs[i] = process_image(combined_imgs[i], IMG_SIZE)

    #images now all have the same size, so we should be able to convert to a np array
    combined_imgs = np.stack(combined_imgs)
        
    #healthy will be given label of 1, infected 0
    #combined_imgs was concatenated with healthy at back, 
    #so can just count up the length of each subset, 
    #labelling first with a 0 then a 1
    labels = []
    
    for i in range(IMG_COUNT[0]):
            labels.append(0)

    for i in range(IMG_COUNT[1]):
            labels.append(1)
    labels = np.array(labels)
     
    if __name__ == '__main__':       
        print("")
        print("Check x, y have compatible shape: ")
        print("Image Shape: {}".format(combined_imgs.shape))
        print("Label Shape: {}".format(labels.shape))
        print("")
    
        elapsed = round(time.time() - start_time)
        print("-- Data processed in {} seconds --".format(elapsed))
    
    return combined_imgs, labels
        
  
def compile_model(model, conv_layer_num, conv_node_num, dense_layer_num):
    
    #One hidden convolutional layer
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)
    print("Input Shape: ", input_shape)
    
    model.add(keras.layers.Conv2D(conv_node_num, kernel_size=5, activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    
    #iterate to conv_layer_num - 1 because there
    #must be at least one conv layer as an input layer
    for a in range(conv_layer_num - 1):
        model.add(keras.layers.Conv2D(conv_node_num, kernel_size=5, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=2))
    
    model.add(keras.layers.Flatten())
    for a in range(dense_layer_num):
        model.add(keras.layers.Dense(128, activation='relu'))
        
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    #compile parameters
    
    loss_fct = keras.losses.binary_crossentropy
    
    model.compile(loss=loss_fct, optimizer='adam', metrics=["accuracy"])

def compare_models(conv_layers, conv_nodes, dense_layers):
    #this method iterates through three passed arrays corresponding to different architectures
    #compares potential models by test accuracy, then returns the optimal values
    
    #load data
    #healthy_imgs, infected_imgs, IMG_COUNT = load_data(DUMMY_HEALTHY_PATH, DUMMY_INFECTED_PATH)
    healthy_imgs, infected_imgs, IMG_COUNT = load_data(HEALTHY_PATH, INFECTED_PATH)
    
    #return np arrays of images and labels
    combined_imgs, labels = process_data(healthy_imgs, infected_imgs, IMG_COUNT, IMG_SIZE)
    
    #split combined data into 80% train/20% test sets
    #sets contains x_train, x_test, y_train, y_test in that order
    sets = [0,0,0,0]
    sets[0], sets[1], sets[2], sets[3] = tts(combined_imgs, labels, test_size=0.2)
    
    current_best_score = 0
    current_best_model = []

    for conv_layer_num in conv_layers:
        for conv_node_num in conv_nodes:
            for dense_layer_num in dense_layers:
                #this will train every possible combination of input parameters
                
                model = keras.models.Sequential()
                model_name = "{} conv layers, {} conv nodes, {} dense layers".format(conv_layer_num, conv_node_num, dense_layer_num)
                
                tfboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(model_name))
                
                print("Training model: {}".format(model_name))
                #print("Shapes of Input: X - {}, Y - {}".format(sets[0].shape, sets[2].shape))
                
                compile_model(model, conv_layer_num=conv_layer_num, conv_node_num=conv_node_num, dense_layer_num=dense_layer_num)
                
                start_time = time.time()
                model.fit(sets[0], sets[2], epochs=10, callbacks=[tfboard])
                elapsed = round(time.time() - start_time)
                print("Model trained in {} secondss".format(elapsed))
                
                model_score = model.evaluate(sets[1], sets[3], batch_size = 64)
                
                print("")
                print("Accuracy: {}%".format(int(model_score[1]* 100)))
                
                print("Loss: {}".format(round(model_score[0], 2)))
                print("")
                if model_score[1] >= current_best_score:
                    print("New current best: {}".format(model_name))
                    current_best_score = model_score[1]
                    current_best_model = [conv_layer_num, conv_node_num, dense_layer_num]
    
    return current_best_model

def train_final_model(shape):
    #Uses the given shape to train a model on all available data
    
    #load data
    healthy_imgs, infected_imgs, IMG_COUNT = load_data(HEALTHY_PATH, INFECTED_PATH)
    
    #return np arrays of images and labels
    combined_imgs, labels = process_data(healthy_imgs, infected_imgs, IMG_COUNT, IMG_SIZE)
    
    model = keras.models.Sequential()
    
    compile_model(model, shape[0], shape[1], shape[2])
    
    model.fit(combined_imgs, labels, epochs=10)

    keras.models.save_model(model, "Best_Model")

if __name__ == "__main__":
    
    
    convs = [1, 2, 3] #numbers of conv layers in different tests
    nodes = [16, 32]  #numbers of conv layer nodes
    dense = [0, 1]    #numbers of dense layers
    
    #compare_models compares the basic structure of different architectures given three
    #arrays of the different configurations to be tested
    best_architecture = compare_models(conv_layers=convs, conv_nodes=nodes, dense_layers=dense)
    
    train_final_model(best_architecture)
    
    
    
    
    





    





