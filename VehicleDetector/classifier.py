from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle


class Classifier(object):
    """
        Classifier model which is used to classify the feature vector 
        into car and non-car
    """
    def __init__(self):
        # self.clf = SVC(probability=True,verbose=True)
        self.clf = RandomForestClassifier(
            n_estimators=150, max_features="auto", min_samples_leaf=4)

    def train(self, X_train, y_train, X_test, y_test):
        self.clf.fit(X_train, y_train)
        print("[accuracy] {}".format(self.clf.score(X_test, y_test)))
        self.save_classifier()

    def predict(self, inputX):
        pred_proba = self.clf.predict_proba(inputX)
        return pred_proba

    def save_classifier(self):
        pickle.dump(self.clf, open('rf_model.pkl', 'wb'))
        print("[INFO] classifier saved")

    def load_classifier(self):
        self.clf = pickle.load(open('rf_model.pkl', 'rb'))
