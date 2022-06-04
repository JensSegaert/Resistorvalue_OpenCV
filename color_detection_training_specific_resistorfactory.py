#########################################################################################################################
""""
Author: Jens Segaert
Date: 03/06/2022
"""
#########################################################################################################################

# Import Libraries
import os.path
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy
import matplotlib.pylab as plt
import cv2
import numpy as np
from PIL import Image
import os.path


path = os.path.abspath(os.getcwd()) + "\images_training_specific_resistorfactory_no-artificial-lighting"
dirs = os.listdir(path)

def training_clustering(path):
    """
       Take all pictures in directory 'images_training_specific_resistorfactory_no-artificial-lighting'
       --> take mean BGR pixel value of each picture
       --> cluster all mean BGR-values of each picture
       """
    BGR_list = []
    BR_list = []
    for item in dirs:
        fullpath = os.path.join(path,item)
        if os.path.isfile(fullpath):
            img = np.array(Image.open(fullpath))
            print('foto')

            # print name_picture:: name_picture = string of fullpath substracted from path string to get name of image itself
            name_picture = fullpath.replace(path, '')

            # remove '\' from image name just for printing
            name_picture = name_picture.replace('\\', '')
            print(name_picture)

            # Read image
            BGR_training_img = cv2.imread(fullpath)

            # Mean pixel value of image (BGR)
            avg_color_per_row = numpy.average(BGR_training_img, axis=0)
            avg_color = numpy.average(avg_color_per_row, axis=0)

            print('mean_BGR_value_picture')
            print(avg_color)
            BGR_list.append(list(avg_color))

            # Make also BR list used for 2D plot of blue and red (green not involved).
            avg_color_BR = avg_color.copy()
            avg_color_BR = list(avg_color_BR)

            # Delete green value from BGR-list and we have BR-list for mean value of each image
            del avg_color_BR[1]
            BR_list.append(avg_color_BR)


            print('BGR_list for now')
            print(BGR_list)
            print('BR_list for now')
            print(BR_list)

    # cluster mean BGR-values images
    # n_clusters is the number of clusters you want to use to classify your data
    kmeans_BGR = KMeans(n_clusters=10, random_state=0).fit(BGR_list)

    # See the labels with:
    print('labels_kmeans')
    print(kmeans_BGR.labels_)

    # See were the centres of your clusters are (BGR)
    print(kmeans_BGR.cluster_centers_)

    list_cluster_centers_BGR = kmeans_BGR.cluster_centers_
    print('list_cluster_centers')
    print(list_cluster_centers_BGR)

    """Only for plotting BR-graph!"""
    # we can now choose to plot clusters. This is going to be a 2D-plot given the Hue and Value of the HSV-values
    # cluster HV-values
    kmeans_BR = KMeans(n_clusters=10, random_state=0).fit(BR_list)

    # See the labels with:
    print('kmeans_BR labels')
    print(kmeans_BR.labels_)

    # See where the centres of your clusters are (BR):
    print(kmeans_BR.cluster_centers_)
    list_cluster_centers = list(kmeans_BR.cluster_centers_)
    print('list_cluster_centers')
    print(list_cluster_centers)
    centroids_BR = kmeans_BR.cluster_centers_



    # Make list of labels
    list_labels = list(kmeans_BGR.labels_)
    print('list_labels')
    print(list_labels)

    label0_list = []
    label1_list = []
    label2_list = []
    label3_list = []
    label4_list = []
    label5_list = []
    label6_list = []
    label7_list = []
    label8_list = []
    label9_list = []

    # Bundle all labels to plot with their color
    for h in range(0,len(list_labels)):
         if list_labels[h] == 0:
             label0_list.append(BGR_list[h])
         if list_labels[h] == 1:
             label1_list.append(BGR_list[h])
         if list_labels[h] == 2:
             label2_list.append(BGR_list[h])
         if list_labels[h] == 3:
             label3_list.append(BGR_list[h])
         if list_labels[h] == 4:
             label4_list.append(BGR_list[h])
         if list_labels[h] == 5:
             label5_list.append(BGR_list[h])
         if list_labels[h] == 6:
             label6_list.append(BGR_list[h])
         if list_labels[h] == 7:
             label7_list.append(BGR_list[h])
         if list_labels[h] == 8:
             label8_list.append(BGR_list[h])
         if list_labels[h] == 9:
             label9_list.append(BGR_list[h])

    for i in range(0,len(label0_list)):
        plt.scatter(label0_list[i][0], label0_list[i][1], color='Green')
    for i in range(0, len(label1_list)):
        plt.scatter(label1_list[i][0], label1_list[i][1], color='Silver')
    for i in range(0,len(label2_list)):
        plt.scatter(label2_list[i][0], label2_list[i][1], color='Orange')
    for i in range(0,len(label3_list)):
        plt.scatter(label3_list[i][0], label3_list[i][1], color='Gray')
    for i in range(0,len(label4_list)):
        plt.scatter(label4_list[i][0], label4_list[i][1], color='Red')
    for i in range(0,len(label5_list)):
        plt.scatter(label5_list[i][0], label5_list[i][1], color='Yellow')
    for i in range(0,len(label6_list)):
        plt.scatter(label6_list[i][0], label6_list[i][1], color='Black')
    for i in range(0,len(label7_list)):
        plt.scatter(label7_list[i][0], label7_list[i][1], color='Purple')
    for i in range(0,len(label8_list)):
        plt.scatter(label8_list[i][0], label8_list[i][1], color='Blue')
    for i in range(0,len(label9_list)):
        plt.scatter(label9_list[i][0], label9_list[i][1], color='Brown')

    # plt.scatter(centroids_BR[:, 0], centroids_BR[:, 1], s=80, color='Black')
    plt.legend()
    # Displaying the title
    plt.title("BR-plot")

    # Display x- and y-axis
    plt.xlabel('B')
    plt.ylabel('R')
    plt.show()


# Call function
training_clustering(path)
