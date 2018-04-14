'''
description: perform PCA, and k-means clustering
author: Alexander Huang
'''

import csv
import pprint
import matplotlib.pyplot as plt
import numpy
from sklearn.cluster import KMeans
def readcsv(filename):
    #col in the csv file, all the attributes
    variables = ["ID", "Milk", "PetFood", "Veggies", "Cereal"
        , "Nuts", "Rice", "Meat", "Eggs", "Yogurt", "Chips"
        , "Cola", "Fruit"]
    #stores the rest of the info
    data = [[]]*len(variables)

    #initial the data with empty list to store the data for each variable
    #for var_index in range(0, len(variables)):
     #   data.append([])

    #open the file
    for variable in range(0, len(variables)):
        data[variable] = []

    with open(filename, 'r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        #reader = csv.DictReader(open("people.csv"))
        for row in reader:
            for var_index in range(0, len(variables)):
                #store the info in to data row by row, varaiables[var_index] == col in the csv files
                data[var_index].append(int(row[variables[var_index]].strip()))

        csvfile.close()

    return data, variables

#normalize a array
def normalize(eignvalue):
    sum = 0
    for v in eignvalue:
        sum += v

    normal_eignvalue = [0]*len(eignvalue)
    for v in range(0, len(eignvalue)):
        normal_eignvalue[v] = round(eignvalue[v] / sum, 3)

    return normal_eignvalue

def main():
    data, variables = readcsv("HW_AG_SHOPPING_CART_v5121.csv")
    #delete ID
    del data[0]
    #get covariance matrix
    cov_data = numpy.cov(data)
    #get eigenvalue, and eigenvector
    eignvalue, eignvector = numpy.linalg.eig(cov_data)
    print("this is eignvalue: \n", eignvalue)
    #normalizew the eignvalue, and sorted
    norm_eignvalue = normalize(eignvalue)
    norm_eignvalue = sorted(norm_eignvalue, reverse=True)
    print("Normalize eignvalue: \n", norm_eignvalue)
    #get two eignvector that has the largest eigenvalue
    eignvector_first2 = [0]*2
    vector1 = [0] * 12
    vector2 = [0] * 12
    print("First two eignvectors: ")
    count = 0
    for d in eignvector:
        print(round(d[0], 2), ", ", end='')
        vector1[count] = round(d[0], 2)
        count += 1

    print()
    count = 0
    for d in eignvector:
        print(round(d[1], 2), ", ", end='')
        vector2[count] = round(d[1], 2)
        count += 1
    print()
    eignvector_first2[0] = vector1
    eignvector_first2[1] = vector2

    #calculate the sum of the eigenvalue for each position of eigenvalue, and print it
    id = [0] * 100
    cumalitive_sum = [0]*12
    sum = 0
    for egin_val in range(0, len(norm_eignvalue)):
        sum += norm_eignvalue[egin_val]
        cumalitive_sum[egin_val] = sum

    #dot product, projection on to eigenvector
    projection = numpy.dot(eignvector_first2, data)
    #print(projection)

    #kmeans clustering
    kmeans_data = [0]*100
    #reorganize data
    for d in range(0, len(data[0])):
        lst = []
        for d2 in range(0, len(data)):
            lst.append(data[d2][d])
        kmeans_data[d] = lst
    #cluster of 3
    kmeans = KMeans(n_clusters= 3).fit(kmeans_data)
    cluster_center = kmeans.cluster_centers_ #center of each cluster
    print("12 demensions: \n", cluster_center)

    cluster_center = numpy.array(cluster_center)
    cluster_center.transpose()

    eignvector_first2 = numpy.array(eignvector_first2)
    eignvector_first2 = eignvector_first2.transpose()
    #project back on to the eigenvector
    proj = numpy.dot(cluster_center, eignvector_first2)
    print("2 demensions: \n", proj)
    '''
    #print sum calc
    x = [1,2,3,4,5,6,7,8,9,10,11,12]
    plt.axis([0, 12, 0, 1])
    plt.plot(x, cumalitive_sum, ".")
    plt.title("Change In Sum of Eignvalue")
    plt.ylabel("Sum of Eignvalue")
    plt.xlabel("Demension")
    plt.show()
    '''
    #print cluster
    plt.axis([-15, 10, -10, 15])
    plt.plot(projection[0], projection[1], ".")
    plt.title("Clustering")
    plt.ylabel("Amount #2")
    plt.xlabel("Amount #1")
    plt.show()

main()