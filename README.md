<!-- ## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.



**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
 -->

### Writeup / README

---

<!-- #### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.   -->

<!-- You're reading it! -->

<div style="text-align:center"><img src=./assets/final.gif width="600" height="400"></div>


---

### Code architecture
```bash
                                            main.py
                                                |
                                                |
                                            pipeline.py
                                                |                   VehicleDetector
               |----------------------------------------------------------------------|
               |   _____________________________|__________________________           |   
               |   |                    |                |                |           |
               |   data_handler.py  features.py     classifier.py    visualizer.py    |
               |                                                                      |   
               |----------------------------------------------------------------------|
```


to run the code  pipeline, uncomment `process_video` in `main.py`
```python
if __name__ == '__main__':
    # unit_tests()
    # train_pipeline()
    process_video() 

```

and execute

```
python main.py
```



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

<!-- The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`: -->


<div style="text-align:center">
    <span style="text-align:center;">____CAR</span>
    <span>|</span>
    <span style="text-align:center;">NON-CAR</span>
</div>
<div style="text-align:center">
    <img src=./assets/image0000.png width="200" height="200">
    <img src=./assets/image1000.png width="200" height="200">
</div>
<div style="text-align:center">
    <img src=./assets/image0001.png width="200" height="200">
    <img src=./assets/image1001.png width="200" height="200">
</div>
<div style="text-align:center">
    <img src=./assets/image0002.png width="200" height="200">
    <img src=./assets/image1002.png width="200" height="200">
</div>
<div style="text-align:center">
    <img src=./assets/image0003.png width="200" height="200">
    <img src=./assets/image1003.png width="200" height="200">
</div>
<div style="text-align:center">
    <img src=./assets/image0004.png width="200" height="200">
    <img src=./assets/image1004.png width="200" height="200">
</div>



---

For every loaded image

<div style="text-align:center"><img src=./assets/car.png width="600" height="400"></div>

The color space of the image is changed using 
```python
def convert_color_space(self,image):
    return cv2.cvtColor(image,self.color_space)
```

<div style="text-align:center"><img src=./assets/changed_color_space.png width="600" height="400"></div>

and the HOG for each channel is computed and concatenated to form a feature vector
```python
 def get_HOG(self,image):
    """
        HOG of every channel of given image is compuuted and
        is concatenated to form one single feature vector
    """
    feat_ch1 = hog(image[:,:,0], 
                        orientations= self.orientations , 
                        pixels_per_cell= self.pixels_per_cell , 
                        cells_per_block= self.cells_per_block,
                        visualise=False)
    feat_ch2 = hog(image[:,:,1], 
                        orientations= self.orientations , 
                        pixels_per_cell= self.pixels_per_cell , 
                        cells_per_block= self.cells_per_block,
                        visualise=False)
    feat_ch3 = hog(image[:,:,2], 
                        orientations= self.orientations , 
                        pixels_per_cell= self.pixels_per_cell , 
                        cells_per_block= self.cells_per_block,
                        visualise=False)
    return np.concatenate((feat_ch1, feat_ch2, feat_ch3))
```

<div style="text-align:center"><img src=./assets/car_hog.png width="600" height="400"></div>

The feature extraction and preprocessing part is handled by `FeatureDetector` class defined in `features.py`
<!-- ![alt text][image2] -->

#### 2. Explain how you settled on your final choice of HOG parameters.

The parameter of HOG were finalized using trial and error method
```python
        self.color_space = cv2.COLOR_RGB2YCrCb
        self.orientations = 16
        self.pixels_per_cell = (12,12)
        self.cells_per_block = (2,2)
        self.image_size = (32,32)
        self.color_feat_size = (64,64)
        self.no_of_bins = 32

        self.color_features = False
        self.spatial_features = False
        self.HOG_features = True
```

It was observed that the color_features and spatial_features don't add much of a significance and rather slow down the computation. Thus only HOG was used

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I found that Random forest classifier gave better accuracy and process time than SVM classifier so I used `RandomForestClassifier` from `sklearn.ensemble` 
```
[accuracy] 0.9583333333333334

```

The following is the feature vector which was used to train the classifier. I created a classifier class which had methods :
```python
def train(self, X_train, y_train, X_test, y_test):
    ...
def predict(self, inputX):
    ...
def save_classifier(self):
    ...
def load_classifier(self):
    ...
```
<div style="text-align:center"><img src=./assets/feature_vect.png width="300" height="300"></div>


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

<!-- I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3] -->
```python
scale_factors = [(0.4,1.0,0.55,0.8,64),
                (0.4,1.0,0.55,0.8,96),
                (0.4,1.0,0.55,0.9,128),
                (0.4,1.0,0.55,0.9,140),
                (0.4,1.0,0.55,0.9,160),
                (0.4,1.0,0.50,0.9,192)]
```

```python
window_1 = self.slide_window(window_image,
                            x_start_stop=[int(scale_factor[0]*width), 
                                            int(scale_factor[1]*width)], 
                            y_start_stop=[int(scale_factor[2]*height), 
                                            int(scale_factor[3]*height)],
                            
                            xy_window=( scale_factor[4], 
                                        scale_factor[4]), 
                            xy_overlap=(0.5, 0.5))
```

Following is the result of sliding windows on the image

<div style="text-align:center"><img src=./assets/window_1.png width="800" height="500"></div>
<div style="text-align:center"><img src=./assets/window_2.png width="800" height="500"></div>
<div style="text-align:center"><img src=./assets/window_3.png width="800" height="500"></div>



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used the above mentioned parameter and window sizes to obtain features. I then created a headmap of the bounding boxes predicted by the classifier, thresholded it and then created blobs to find the final bounding boxes.

<div style="text-align:center"><img src=./assets/multiple_detected_boxes.png width="1200" height="600"></div>
<div style="text-align:center"><img src=./assets/cars_heatmap.png width="900" height="700"></div>
<div style="text-align:center"><img src=./assets/labels.png width="900" height="700"></div>
<div style="text-align:center"><img src=./assets/final_car.png width="900" height="700"></div>

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=epJj0KiRnco&feature=youtu.be)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I took the predictions which had probability above a given threshold and created a heatmap from their bounding boxes. I then made the heatmap binary thresholding it. To find blobs I used `scipy.ndimage.measurements.label()` . To track the object and reduce affect of false positives, I stored the previous heatmap and added a scaled version of it to the next heatmap and increased the heapmap threshold. thus the heat in the area where previously car was detected would be higher than original and can easily be thresholded. 

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:


<div style="text-align:center">
    <img src=./assets/f1.png width="250" height="250">
    <img src=./assets/h1.png width="250" height="250">
    <img src=./assets/v1.png width="250" height="250">
</div>
<div style="text-align:center">
    <img src=./assets/f2.png width="250" height="250">
    <img src=./assets/h2.png width="250" height="250">
    <img src=./assets/v2.png width="250" height="250">
</div>
<div style="text-align:center">
    <img src=./assets/f3.png width="250" height="250">
    <img src=./assets/h3.png width="250" height="250">
    <img src=./assets/v3.png width="250" height="250">
</div>
<div style="text-align:center">
    <img src=./assets/f4.png width="250" height="250">
    <img src=./assets/h4.png width="250" height="250">
    <img src=./assets/v4.png width="250" height="250">
</div>
<div style="text-align:center">
    <img src=./assets/f5.png width="250" height="250">
    <img src=./assets/h5.png width="250" height="250">
    <img src=./assets/v5.png width="250" height="250">
</div>
<div style="text-align:center">
    <img src=./assets/f6.png width="250" height="250">
    <img src=./assets/h6.png width="250" height="250">
    <img src=./assets/v6.png width="250" height="250">
</div>



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This implementation is very slow and requires a lot of parameter tuning. A deep learning approach would be faster and more robust as forward pass in architectures like faster RCNN and YOLO is in microseconds. There would be no need to choose features manually. The outliers in the deep learning approach can also be removed by non-maximal suppression,   

