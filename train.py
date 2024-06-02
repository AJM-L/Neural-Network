import numpy as np
from os.path  import join
import random
import matplotlib.pyplot as plt

from AJsNetwork import Network
from Loader import MnistDataloader


#
# Verify Reading Dataset via MnistDataloader class
#
#%matplotlib inline


#
# Set file paths based on added MNIST Datasets
#
input_path = '/Users/ajmatheson-lieber/Desktop/NeuralNetFinal/Datafile/'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1
    plt.show()

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
x_train_formatted = x_train[:]
x_test_formatted =  x_test[:]
for i in range(len(x_train)):
    ex = np.concatenate(x_train[i])
    ex = np.reshape(ex, (784, 1))
    x_train[i] = ex
for i in range(len(x_test)):
    ex = np.concatenate(x_test[i])
    ex = np.reshape(ex, (784, 1))
    x_test[i] = ex
"""
y_test = list(y_test)
for i in range(len(y_test)):
    ex = np.zeros((10, 1))
    ex[y_test[i]][0] = 1
    y_test[i] = ex
"""
y_non_vector = y_train
y_train = list(y_train)
for i in range(len(y_train)):
    ex = np.zeros((10, 1))
    ex[y_train[i]][0] = 1
    y_train[i] = ex

#
# Show some random training and test images 
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train_formatted[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_non_vector[r]))    

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test_formatted[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

#shows a sample of digits. Close pop-up window to let the network training begin.
show_images(images_2_show, titles_2_show)

# Trains a Network with two hidden layers of 200 and 80 nodes respectively fro 10 epochs
# Network can be called successively with MyNet variable 
# Usually yields around 75% accuracy with this setup (random would yield about 10% accuracy).
# Accuracy can go much higher with necassary computational power. The best I have gotten was around 95%.
MyNet = Network([784, 200, 80, 10])
MyNet.SGD(list(zip(x_train, y_train)), 10, 100, 1.5)
print("Network classfied Mnist set at " + str(MyNet.evaluate(list(zip(x_test, y_test)))) + "% accuracy.")
