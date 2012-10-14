#### INTRODUCTION

Two R script is implement based on KNN algorithm.

+ __knn_1d_demo__ : Estimating probability distribution from 10,000 samples which are drawn from N(5.,3.)

+ __knn_classify__ : A KNN based classifier used to classify hand-writing data.

#### DEMO

This script use knn to estimate probability distribution. 10,000 samples are drawn from N(5.,3.), Histogram is shown below.

![hist](https://raw.github.com/Oneplus/anothermlkid/master/knn/image/knn_1d_demo_hist.png)

and here comes the estimated distribution

![estimate_pd](https://raw.github.com/Oneplus/anothermlkid/master/knn/image/knn_1d_demo_1.png)

#### CLASSIFIER

This script use knn to classify handwritten data, which is available in follow site [UCI Machine Learning Repository: Semeion Handwritten Digit Data Set](http://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit)

The data is a 16*16 0-1 matrix which illustrate handwritten photograph, like

```
      ........
     ...... ..
    ......   ..
   .....    ....
   .... .......
   ... ..... ..
  ........   ..
  ......    ..
 ......     ..
 ....       ..
 ...      ...
...       ..
....    ....
....   ...
. .......
   ....
0
```

Fortuately, much work of graph transformation is done before downloading the data.