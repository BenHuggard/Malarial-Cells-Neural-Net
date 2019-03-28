#this script does two things:
#1)tests the model trained in the other script with a selection of hand-picked
#  images taken from Google Images
#2)provides a function which accepts a path to an image, and prints to console
#  its diagnosis of that image

import tensorflow as tf
import ModelTrainer as MT
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

loaded_model = tf.keras.models.load_model("Best_Model")

DIRECTORY = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = [40, 46] #get dimensions to resize to when training model, set them here

def diagnose_image(image):
    #pass an np array of image, return a diagnosis
    
    #model expects a 4D array sized to the minimum image dims in the training data
    #minimum dims are found when training model and must be entered above
    image = np.expand_dims(image, axis=0)
    
    prediction = loaded_model.predict(image)
    
    diagnosis = "Infected" if prediction[0][0] < 0.5 else "Healthy"
    
    return diagnosis

def test_web_data():
    #load data taken manually from google images for testing
    HEALTHY_PATH = DIRECTORY + "/Web Test Images/Uninfected"
    INFECTED_PATH = DIRECTORY + "/Web Test Images/Parasitized"
    
    print("")
    print("Testing a selection of images taken manually from the web...")
    print("")
    
    healthy_imgs, infected_imgs, IMG_COUNT = MT.load_data(HEALTHY_PATH, INFECTED_PATH)
    combined_imgs, labels = MT.process_data(healthy_imgs, infected_imgs, IMG_COUNT, IMG_SIZE)
    
    label_index = 0
    for image in combined_imgs:
        plt.imshow(np.squeeze(image), cmap='gray')

        plt.show()
        
        cell_status = "Infected" if labels[label_index] == 0 else "Healthy"
        
        print("")
        print("Cell is {}".format(cell_status))
        
        diagnosis = diagnose_image(image)
        print("Model predicts: {}".format(diagnosis))
        
        label_index += 1

def test_image_from_path(path):
    print("")
    print("Testing data at: {}".format(path))
    
    image = Image.open(path)
    image = np.array(image)
    plt.imshow(image)
    plt.show()
    
    IMG_SIZE = [40, 46]
    image = MT.process_image(image, IMG_SIZE)
    
    diagnosis = diagnose_image(image)
    
    print("")
    print("Model predicts: {}".format(diagnosis))
    

test_web_data()

TEST_PATH = DIRECTORY + "/test_individ/test.png"

test_image_from_path(TEST_PATH)









