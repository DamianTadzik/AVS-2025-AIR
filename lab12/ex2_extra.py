import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

from ex1 import hog  # zakładam, że hog() zwraca wektor 3780 elementów

num_samples = 924  # 924 pozytywnych i 924 negatywnych = 1848 przykładów

# def augment_image(img):
#     # Prosta augmentacja: odbicie lustrzane poziome
#     return cv2.flip(img, 1)

def augment_image(img):
    # Odbicie lustrzane
    img_flip = cv2.flip(img, 1)

    # Obrót o losowy kąt z zakresu np. -10 do 10 stopni
    angle = np.random.uniform(-10, 10)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # warpAffine z wypełnieniem odbiciem krawędzi (border reflect)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    # Możesz też dołożyć rozjaśnienie/ciemnienie:
    alpha = np.random.uniform(0.9, 1.1)  # kontrast
    beta = np.random.uniform(-10, 10)    # jasność
    adjusted = cv2.convertScaleAbs(rotated, alpha=alpha, beta=beta)
    
    return [img, img_flip, adjusted]  # zwróć listę augmentowanych obrazów


def prepare_dataset(num_samples, start_idx=1, augment=False):
    X = []
    y = []
    for i in range(num_samples):
        # Pozytywne próbki
        pos_img = cv2.imread(f'pedestrians/pos/per{i+start_idx:05d}.ppm')
        pos_img = cv2.cvtColor(pos_img, cv2.COLOR_BGR2RGB)
        X.append(hog(pos_img))
        y.append(1)
        if augment:
            pos_aug = augment_image(pos_img)
            if isinstance(pos_aug, list):
                for img in pos_aug:
                    X.append(hog(img))
                    y.append(0)
            else:
                X.append(hog(pos_aug))
                y.append(0)

        # Negatywne próbki
        neg_img = cv2.imread(f'pedestrians/neg/neg{i+start_idx:05d}.png')
        neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)
        X.append(hog(neg_img))
        y.append(0)
        if augment:
            neg_aug = augment_image(neg_img)
            if isinstance(neg_aug, list):
                for img in neg_aug:
                    X.append(hog(img))
                    y.append(0)
            else:
                X.append(hog(neg_aug))
                y.append(0)
    return np.array(X, dtype=np.float32), np.array(y)

if __name__ == '__main__':
    print("Preparing full dataset with augmentation...")
    X, y = prepare_dataset(num_samples, start_idx=1, augment=False)  # 924 * 2 * 2 = 3696 próbek

    print("Running 4-fold cross-validation with GridSearchCV...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=2137)
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=cv, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)

    print(f"\nBest hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%")

    best_clf = grid_search.best_estimator_
    best_clf.fit(X, y)

    joblib.dump(best_clf, 'svm_hog_model_cross_validated.joblib')
    print("\nModel saved as 'svm_hog_model_cross_validated.joblib'")

    # Ewaluacja na całym zestawie (trening + augmentacja)
    print("Evaluating model on full training set...")
    pred = best_clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"Train set - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Train Accuracy: {accuracy_score(y, pred) * 100:.2f}%")
