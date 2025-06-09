import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

from ex1 import hog 

num_train = 100
num_val = 500
TRAIN = False



# Prepare training data
HOG_train = np.zeros((2 * num_train, 3781), dtype=np.float32)
for i in range(num_train):
    pos_img = cv2.imread(f'pedestrians/pos/per{i+1:05d}.ppm')
    pos_img = cv2.cvtColor(pos_img, cv2.COLOR_BGR2RGB)
    pos_feat = hog(pos_img)
    HOG_train[i, 0] = 1
    HOG_train[i, 1:] = pos_feat

    neg_img = cv2.imread(f'pedestrians/neg/neg{i+1:05d}.png')
    neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)
    neg_feat = hog(neg_img)
    HOG_train[i + num_train, 0] = 0
    HOG_train[i + num_train, 1:] = neg_feat

train_labels = HOG_train[:, 0]
train_data = HOG_train[:, 1:]

import joblib
if TRAIN:
    # Train SVM
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(train_data, train_labels)
    # Save
    joblib.dump(clf, 'svm_hog_model.joblib')
else:
    # Load
    clf = joblib.load('svm_hog_model.joblib')

# Predict on training set
train_pred = clf.predict(train_data)
tn, fp, fn, tp = confusion_matrix(train_labels, train_pred).ravel()
print(f"Train set - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Train Accuracy: {np.mean(train_pred == train_labels) * 100:.2f}%")

# Prepare validation data
HOG_val = np.zeros((2 * num_val, 3781), dtype=np.float32)
for i in range(num_val):
    pos_img = cv2.imread(f'pedestrians/pos/per{i+1+num_train:05d}.ppm')
    pos_img = cv2.cvtColor(pos_img, cv2.COLOR_BGR2RGB)
    pos_feat = hog(pos_img)
    HOG_val[i, 0] = 1
    HOG_val[i, 1:] = pos_feat

    neg_img = cv2.imread(f'pedestrians/neg/neg{i+1+num_train:05d}.png')
    neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)
    neg_feat = hog(neg_img)
    HOG_val[i + num_val, 0] = 0
    HOG_val[i + num_val, 1:] = neg_feat

val_labels = HOG_val[:, 0]
val_data = HOG_val[:, 1:]

# Predict on validation set
val_pred = clf.predict(val_data)
tn, fp, fn, tp = confusion_matrix(val_labels, val_pred).ravel()
print(f"Validation set - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Validation Accuracy: {np.mean(val_pred == val_labels) * 100:.2f}%")

'''If you want to improve further, consider:

Trying different C values or kernels (RBF, polynomial)

Increasing training sample size

Augmenting data (flips, rotations)

Using cross-validation splits for more robust validation
''''
