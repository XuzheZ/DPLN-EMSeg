
## ECBM 4040 Final Project: [Learning-based EM image segmentation]
Our group dealt with the segmentation of electron microscopy (EM) images of neuronal structures using the PyraMid-LSTM model proposed by the [paper](https://arxiv.org/abs/1506.07452). We also achieved a Multi-Dimensional LSTM (MD-LSTM) model for comparison.


### Group: DPLN
* 	Huixiang Zhuang hz2538 Columbia University
* 	Yutao Tang      yt2639 Columbia University
* 	Xuzhe Zhang     xz2778 Columbia University

### Getting Started
These instructions will provide you a guideline for our basic functions as well as how to running on your machine for development and testing purposes.
#### Prerequisites
>python 3.6 (some specific packages you may need)
>>numpy:         pip install numpy
>
>>libtiff:        pip install libtiff
>
>>tensorflow:      pip install tensorflow-gpu
>
>>matplotlib:      pip install matplotlib


>Jupyter Notebook 

#### Introducing the files in project
main.iqynp
>This script is our main jupyter notebook. Implemented our whole project.
>
loaddata.iqynp
>This script is our second jupyter notebook. However, you have to run it first, since it will read the tiff file and convert it to array
>make sure you hace already install libtiff before using it.
./DPLN/pyramidLSTM.py
>
>
./DPLN/Net_PyramidLSTM.py
>
>
./DPLN/MDLSTM.py
>
>
./DPLN/Net_MDLSTM.py
>
>
./DPLN/ProcGenerator.py
>This py file is refered to as a batch generator.
>
./DPLN/ProcessingV2.py
>This py file provides the basic function for image preprocessing, including ROF denoise, Z-score normalization and augmentation.
>
test-volume.tiff, train-volume.tiff, train-label.tiff
>They are the training data, labels and test data. Because it is a little tricky to download them, we provide them in the submission
>
./model
>Save the trained model in this folder
mainOut256.iqynb
>this is the automated produced version when we trained our model on the serverl. It record the printout output while training the Pyramid-LSTM, with a batch size 256*256
>
### Running the test
#### 1. Firstly,  _run the loaddata.iqynb_ to read the tiff image.
#### 2. Secondly, Open _**main.iqynb**_  through Jupyter Notebook:
* Run the **first several cells** to finish modules importing, to load the data and labels as well as to implement preprocessing
    * This step will print out the size of data and label
    * This step apply ROF denoise and Z-score to training data and test data, and also apply data augmentation to training data and labels
* For PyramidLSTM training, run the **corresponding cell** (which contains training function)
    * Set up proper batch_size (how many batch used in one step), input size (the size of voxel batch, and max_step (training steps). 
    * The recommendation is: batch_size=1, input_size=[128,128,15], max_step=1000, (256*256 training is very time-consuming, may take about 12 mins for 10 steps)
    * This step will print out the number of net parameters, restored model, and will display the loss for every 10 steps.
* For PyramidLSTM validation, run the **corresponding cell** (which contains validation function)
    * The fifth cell will reshape and save the validation sets remained in the training. It will print out th size of validation set.
    * The sixth cell can load the saved validation set.
    * The seventh cell will prediction for validation sets. And you can run the next two cells to save the prediction to .tiff file.
* For PyramidLSTM test, run the run the **corresponding cell** (which contains test function)
    * This step will use the trained model "model/pyramidnet_size128" to finish the prediction for test data. You can run the next cell to save the prediction result to .tiff file. And upload this 32-bit .tiff file to [2012 ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/) for evaluation.
* Please kindly notice that, in the main.iqynb, some cells have been to converted Markdown, if you find some cell can't run, please kindly try to convert above or below cells to code, and try again. We don't have so many time to clean them. Sorry for any inconvenience, we very appreciate your understanding.


####  Or: Run _**main.iqynb**_ from terminal
* You can also run the script from terminal, after enter your proper environment, and cd to the proper folder by 
>cd ./project 
* and input:
>$ jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=600 main.ipynb 
>

### Our data set
In our project, the data set we uesd is The Drosophila first instar larva ventral nerve cord (VNC, [Cardona et al., 2010, 2012](https://www.ini.uzh.ch/~acardona/trakem2.html)) , provided by [2012 ISBI Challenge: Segmentation of neuronal structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/). It contains 30 consecutive images as the training set with the ground truth 2D segmentation as training labels, as well as 30 consecutive images as the testing set. The microcube measures 2 x 2 x 1.5 microns approx, with a resolution of 4x4x50 nm/pixel. ![**Templete training data and labels**](http://brainiac2.mit.edu/isbi_challenge/sites/default/files/Challenge-ISBI-2012-Animation-Input-Labels.gif)<center>Templete training data and labels (image from [ISBI](http://brainiac2.mit.edu/isbi_challenge/))</center>
This dataset is not completely public. If you want to get access to the dataset, you have to [register](http://brainiac2.mit.edu/isbi_challenge/user/register), to login and then you can download the data. Since the dataset is relative small-size: 3 .tiff file, including 30 slices of 512*512 TEM images, with a size about 180MB. So we will provide the dataset in our submission.

### Evaluation
The evaluation of the test result is based on two empirical metrics: **Foreground-restricted Rand Scoring after border thinning**: $V^{Rand}$ and **Foreground-restricted Information Theoretic Scoring after border thinning**: $V^{Info}$.
Since the ground-truth of test data is not provided publicly, to evaluate the test result, you have to register and login the [2012 ISBI Challenge](http://brainiac2.mit.edu/isbi_challenge/) website and then upload the result. The evaluation is manually implemented by the logistics of ISBI, so it may take some time.

### Our result
![test image](https://github.com/XuzheZ/DPLN-EMSeg/blob/master/images/test-volume.gif?raw=true)<center>Test data </center>![prediction](https://github.com/XuzheZ/DPLN/blob/master/prediction.GIF?raw=true)<center>Prediction </center>

