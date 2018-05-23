import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


class DataHandler(object):
    """
            Class which handles data related methods
    """

    def __init__(self, BASE_PATH="."):
        self.BASE_PATH = BASE_PATH

    def load_data(self):
        """
            using GLOB all the directories are searched for 
            required images and list of paths are updated
        """
        self.test_images = glob.glob("./test_images/*.jpg")
        self.vehicles = glob.glob("./vehicles/**/*.png")
        self.non_vehicles = glob.glob("./non-vehicles/**/*.png")

        print("[test_images] {}".format(len(self.test_images)))
        print("[vehicles] {}".format(len(self.vehicles)))
        print("[non_vehicles] {}".format(len(self.non_vehicles)))

    def prepare_trainable_data(self, _features, labels):
        """
            The car and non-car data is loaded here and split into
            train and test data. The scalar fit during this process is   
        """
        X_scaler = StandardScaler().fit(_features)
        scaled_X = X_scaler.transform(_features)

        pickle.dump(X_scaler, open('X_scaler.pkl', 'wb'))

        X_train, X_test, y_train, y_test = train_test_split(scaled_X, labels,
                                                            test_size=0.2)
        print('[Training data]{} [Training labels]{}'.format(X_train.shape, 
                                                            y_train.shape))
        print('[Test data]{} [Test labels]{}'.format(X_test.shape, 
                                                    y_test.shape))
        return (X_train, X_test, y_train, y_test)

    def load_scalar(self):
        self.X_scaler = pickle.load(open('X_scaler.pkl', 'rb'))

    def scale_vector(self,vector):
        return self.X_scaler.transform(vector)

