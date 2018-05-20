from sklearn.svm import SVC
import numpy as np
import pickle

class Classifier(object):
	def __init__(self):
		self.clf = SVC(probability=True,verbose=True)
	
	def train(self,X_train,y_train,X_test,y_test):
		self.clf.fit(X_train,y_train)
		print("[accuracy] {}".format(self.clf.score(X_test,y_test)))
		self.save_classifier()
	
	def predict(self):
		pass

	def save_classifier(self):
		pickle.dump(self.clf, open('svm_model.pkl', 'wb'))
		print("[INFO] classifier saved")

	def load_classifier(self):
		self.clf = pickle.load(open('svm_model.pkl', 'rb'))