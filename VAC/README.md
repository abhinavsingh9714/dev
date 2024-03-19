Video_Action_classification
==============================
Video-Action-Classifier
Video based Action Classification using LSTM

Dataset: This dataset consists of labelled videos of 6 human actions (walking, jogging, running, boxing, hand waving and hand clapping) performed several times by 25 subjects in four different scenarios: outdoors s1, outdoors with scale variation s2, outdoors with different clothes s3 and indoors s4 as illustrated below.

![alt text](image-3.png)![alt text](image-4.png)

All sequences were taken over homogeneous backgrounds with a static camera with 25fps frame rate. The sequences were downsampled to the spatial resolution of 160x120 pixels and have a length of four seconds in average. In summary, there are 25x6x4=600 video files for each combination of 25 subjects, 6 actions and 4 scenarios. For this project we have randomly selected 20% of the data as test set.

Dataset source: https://www.csc.kth.se/cvap/actions/

Methodology:

When performing image classification, we input an image to our CNN; Obtain the predictions from the CNN; Choose the label with the largest corresponding probability

Since a video is just a series of image frames, in a video classification, we Loop over all frames in the video file;
![alt text](image.png) 
For each frame, pass the frame through the CNN; Classify each frame individually and independently of each other; Choose the label with the largest corresponding probability; Label the frame and write the output frame to disk

Background: The CNN LSTM architecture involves using Convolutional Neural Network (CNN) layers for feature extraction on input data combined with LSTMs to support sequence prediction.

CNN LSTMs were developed for visual time series prediction problems and the application of generating textual descriptions from sequences of images (e.g. videos). Specifically, the problems of:

Activity Recognition: Generating a textual description of an activity demonstrated in a sequence of images Image Description: Generating a textual description of a single image. Video Description: Generating a textual description of a sequence of images. Applications: Applications such as surveillance, video retrieval and human-computer interaction require methods for recognizing human actions in various scenarios. In the area of robotics, the tasks of autonomous navigation or social interaction could also take advantage of the knowledge extracted from live video recordings. Typical scenarios include scenes with cluttered, moving backgrounds, nonstationary camera, scale variations, individual variations in appearance and cloth of people, changes in light and view point and so forth. All of these conditions introduce challenging problems that can be addressed using deep learning (computer vision) models.

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── training.py    <- contains the main script  
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
 
 Results: 
 Models: 
 1. Hyperparameter tuned model 
 Model.summary()
_________________________________________________________________________________
| Layer (type)                          |  Output Shape           |   Param #   | 
|=======================================|=========================|=============|
| time_distributed_30 (TimeDistributed) | (None, 10, 78, 78, 320) |  3200       |
|                                       |                         |             |                                       
| time_distributed_31 (TimeDistributed) | (None, 10, 78, 78, 320) |  1280       |
|                                       |                         |             |
| time_distributed_32 (TimeDistributed) | (None, 10, 39, 39, 320) |  0          |
|                                       |                         |             |                                     
| time_distributed_33 (TimeDistributed) | (None, 10, 39, 39, 128) |  368768     |
|                                       |                         |             |                                         
| time_distributed_34 (TimeDistributed) | (None, 10, 39, 39, 128) |  512        |
|                                       |                         |             |                                        
| time_distributed_35 (TimeDistributed) | (None, 10, 19, 19, 128) |  0          |
|                                       |                         |             |                                         
| time_distributed_36 (TimeDistributed) | (None, 10, 19, 19, 128) |  147584     |
|                                       |                         |             |                                       
| time_distributed_37 (TimeDistributed) | (None, 10, 19, 19, 128) |  512        |
|                                       |                         |             |                              
| time_distributed_38 (TimeDistributed) | (None, 10, 9, 9, 128)   |  0          |
|                                       |                         |             |                               
| time_distributed_39 (TimeDistributed) | (None, 10, 10368)       |  0          |
|                                       |                         |             |                                  
| dropout_7 (Dropout)                   | (None, 10, 10368)       |  0          |
|                                       |                         |             |
| lstm_3 (LSTM)                         | (None, 96)              |  4018560    |
|                                       |                         |             |
| dense_7 (Dense)                       | (None, 64)              |  6208       |
|                                       |                         |             |
| dropout_8 (Dropout)                   | (None, 64)              |  0          |
|                                       |                         |             |
| dense_8 (Dense)                       | (None, 6)               |  390        |
_________________________________________________________________________________
Total params: 4547014 (17.35 MB)
Trainable params: 4545862 (17.34 MB)
Non-trainable params: 1152 (4.50 KB)
_________________________________________________________________
Model performance: Accuracy on Validation set: 85.23%
 ![alt text](image-1.png)

2. VGG16 pretrained model
Model.summary()
_____________________________________________________________________________________
| Layer (type)                             |      Output Shape       |     Param #  |
|__________________________________________|_________________________|______________|
| time_distributed_112 (Time Distributed)  | (None, 10, 3, 2, 512)   |     14714688 | 
|                                          |                         |              |                                
| time_distributed_113 (TimeDistributed)   |   (None, 10, 3072)      |       0      |   
|                                          |                         |              |                                     
|lstm_40 (LSTM)                            |     (None, 1200)        |    20510400  | 
|                                          |                         |              |
| dense_81 (Dense)                         |     (None, 1024)        |      1229824 |  
|                                          |                         |              |
| dropout_30 (Dropout)                     |     (None, 1024)        |      0       |  
|                                          |                         |              |
| dense_82 (Dense)                         |      (None, 6)          |       6150   |   
_____________________________________________________________________________________
Total params: 36461062 (139.09 MB)
Trainable params: 21746374 (82.96 MB)
Non-trainable params: 14714688 (56.13 MB)
_________________________________________________________________


Model performance: Accuracy on Validation set: 79.167%
|             | precision  |  recall | f1-score |  support   |
|_____________|____________|_________|__________|____________|
|      boxing |      1.00  |    0.85 |     0.92 |       20.  |
|     running |      0.59  |    0.65 |     0.62 |       20   |
|Handclapping |      0.65  |    0.55 |     0.59 |       20   |
|     jogging |      0.29  |    0.25 |     0.27 |       20   |
|     Walking |      0.67  |    0.70 |     0.68 |       20   |
|  handwaving |      0.54  |    0.70 |     0.61 |       20   |
|_____________|____________|_________|__________|____________|
|    accuracy |            |         |     0.62 |      120   |
|   macro avg |      0.62  |    0.62 |     0.62 |      120   |
|weighted avg |      0.62  |    0.62 |     0.62 |      120   |

![alt text](image-2.png)
