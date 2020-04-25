# Mafat Challenge - Detection and fine grained classification of objects in aerial imagery
This repo is a pilot (pre-release) implementation of the Computer Vision [Mafat Challenge](https://mafatchallenge.mod.gov.il/) ([competition site](https://competitions.codalab.org/competitions/19854)).  
The main differences I'm aware of between this version and the official released-to-public version:  
| This version  | Official version |
| ------------- | ------------- |
| Detection & Classification  | Classification  |
| Classes of Large Vehicle, Small Vehicle, Solar Panel and Utility Pole  | Classes of Large Vehicle and Small Vehicle  |
| 9369 images | 4216 images |

Hence, solving this version is different.  


## Index
* `Detecting And Classifying Objects In Aerial Imagery/` folder contains challenge's data (CSVs) and used to contain the dataset as well.
* `src/` folder contains code for training & predicting the F-RCNN, ResNet-50 networks as well as several more utility scripts.
* [Pilot challenge PDF](https://github.com/CodeJjang/mafat-aerial-challenge-pilot/blob/master/Detecting%20And%20Classifying%20Objects%20In%20Aerial%20Imagery/Detecting%20And%20Classifying%20Objects%20In%20Aerial%20Imagery%20191117.pdf)
* [Final report PDF](https://github.com/CodeJjang/mafat-aerial-challenge-pilot/blob/master/docs/report.pdf)

