import sys, os
import numpy as np
import cv2
import pickle


from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


sys.path.append('../.')

from circularHOGExtractor import circularHOGExtractor
ch = circularHOGExtractor(4,2,4) 

cls0 = './no/'
cls1 = './yes/'

lst0 = [name for name in os.listdir(cls0) if not name.startswith('.')] 
lst1 = [name for name in os.listdir(cls1) if not name.startswith('.')]

nFeats = ch.getNumFields() 
trainData = np.zeros((len(lst0)+len(lst1),nFeats))
targetData = np.hstack((np.zeros(len(lst0)),np.ones(len(lst1))))

i = 0
for imName in lst0:
    sample = cv2.imread(cls0 + imName)
    thisIm = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    trainData[i,:] = ch.extract(thisIm)
    i = i + 1
for imName in lst1:
    sample = cv2.imread(cls1 + imName)
    thisIm = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    trainData[i,:] = ch.extract(thisIm)
    i = i + 1

#clf = svm.SVC()

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=12),algorithm="SAMME",n_estimators=50)
y_pred = clf.fit(trainData,targetData)
pickle.dump(clf, open( "../svmClassifier.p","wb"))
y_pred = clf.predict(trainData)
print("Number of mislabeled points out of a total %d points : %d" % (trainData.shape[0],(targetData != y_pred).sum()))
