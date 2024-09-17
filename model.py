from __future__ import print_function
import numpy as np
from sklearn import neighbors
# from sklearn.model_selection import train_test_split # for splitting data
# from sklearn.metrics import accuracy_score # for evaluating results
import csv
import cv2
from sklearn.svm import SVC

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
def string_data_to_float(dataset):
	for i in range(len(dataset[0]) ):
		str_column_to_float(dataset, i)
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def feature(image,image_binary):
    h, w = image_binary.shape
    sum=[0,0,0]
    count=0
    for i in range(h):
        for j in range(w):
            if image_binary[i][j] == 255:
                sum[0] += image[i][j][0]
                sum[1] += image[i][j][1]
                sum[2] += image[i][j][2]
                count += 1

    sum[0] = sum[0] / count/255
    sum[1] =  sum[1] / count/255
    sum[2] =  sum[2] / count/255
    # print(sum)
    # print(time.time() - t1)
    return sum
def data_read_and_processing(file):
    dataset=load_csv(file)
    string_data_to_float(dataset)
    return dataset
def data_train_model(dataset):
    data_x = np.array([row[0:-1] for row in dataset])
    data_y = np.array([row[-1] for row in dataset])
    return data_x,data_y

if __name__=="__main__":
    #data mẫu
    dataset = data_read_and_processing('data.csv')
    data_x, data_y = data_train_model(dataset)

    #model classification KNN,SVM
    model_SVM = SVC()
    model_SVM.fit(data_x, data_y)
    model=neighbors.KNeighborsClassifier(n_neighbors=5,p=2)
    model.fit(data_x,data_y)

    #test
    image_test=cv2.imread('1.png',1)

    #thresold image
    image_test_hsv=cv2.cvtColor(image_test,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(image_test_hsv)
    _,thresold1=cv2.threshold(v,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #Morphological Operations: openning and clossing
    kernel = np.ones((15, 15), np.uint8)
    openning=cv2.morphologyEx(thresold1,cv2.MORPH_OPEN,kernel)
    closing=cv2.morphologyEx(openning,cv2.MORPH_CLOSE,kernel)
    #show image
    cv2.imshow('v', v)
    cv2.imshow('locnhieu', closing)
    cv2.imshow('web_cam', image_test)
    #predict
    x_test=np.array([feature(image_test,closing)])

    y_pred=model.predict(x_test)
    y_pred1 = model_SVM.predict(x_test)
    print(x_test)
    print(y_pred)
    print(y_pred1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()