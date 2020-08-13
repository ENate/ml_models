## talpasolutions-task


A model to classify the activity performed by a roofbolter.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
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
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
-----------------
### Task:
The solution can be run in two parts. In the ../notebooks folder, we present an initial analysis of the problem. The example in the `lstm_model_new.ipynb`.
The initial pre-processing step involves imputing the 'NaN' values and normalization of the data set. The 'activity' column was also encoded to enable the use of a suitable deep neural network algorithm during training.

 #### The Training Steps and Parameters:
 * In order to run the model, change to the ~/talpa-datascience-task/
 `python train_model.py`
 Some of the main tools required to run the entire project are listed in the `requirements.txt` folder.
 * In the directory structure shown above, both the `make_dataset.py` and `build_features.py` can be used to create a data set and build features respectively.
 * The `visualize.py` file contain examples of some confusion matrices corresponding to the output classes.
 Deep neural networks (Such as stacked layers of LSTMs in this case) are capable of automatic feature selection (especially when a regularization is introduced during training). This can lead to
  the most significant input features being identified from the dataset. Therefore, I chose a recurrent neural network such as an LSTM (to avoid the vanishing gradient problem). 
  The choice of LSTM enables us to pad sequences (representing timestamps) at the same length. I used 100, 200 but training with these do not show any difference in the accuracy.

#### Summary of the Results
The confusion matrices below:
`a=[[[4990,    0],
  [   2,    0]]
  
 [[3808,   26],
  [17, 1141]]

 [[4782,   28],
  [  16,  166]]

 [[4546   20],
  [  20  406]]

 [[3066    7],
  [   9 1910]]

 [[4333   20],
  [  44  595]]

 [[4313   13],
  [  31  635]]]`
 #### The Precision, recall and fscore for each class label are:
precision: [0., 0.9734589,  0.87209302, 0.96542553, 0.99522546, 0.95524691, 0.96940195]
recall: [0., 0.98869565, 0.84269663, 0.94285714, 0.99946723, 0.91703704, 0.96403873]
fscore: [0., 0.98101812, 0.85714286, 0.95400788, 0.99734184, 0.93575208, 0.9667129 ]

### Conclusion
The final accuracy is over 95.458% after a random initialization of the neural network's weights got initialized at the start of the training procedure.

This means that the LSTM model is almost always able to correctly identify the movement and the different states of the Machine. 
We recall that the moving states of the Machine are recorded at each series to classify different activity states. The data sets was formatted  (by
 assigning sequence steps of 64, 128 at different times) but do not affect the overall outcome.
  The predictions are extremely accurate given this small window of context and raw data. 
The data sets were divided into a ratio of 80:20 ratio with and without randomization of the activity column. 
In doing so, no significant disruption was made with respect to the (accuracy) results.
The confusion matrices indicate different True Positives (TF) and False Positives (FP). Meanwhile,
the normalized confusion matrix diagram printed at the end of the training indicates the distribution of the different states of the Machine.

### Further Analysis
Improvements can be made to the different algorithms to investigate the influence of training times, and other performance metrics.
The choice of using the LSTM network in modelling is due to its proven ability in 'being able to recall immediate past data' which is prevalent in
 time series classification/predictions. A case can be made for KNN classifiers, 1-dimensional Convolutional Neural Networks (CNNs) as well. In the latter case,
 a masking layer could be used to ignore the padded time steps. Also, the code can be improved by implementing via the 
 `tf.Estimator` API whereby, the training, prediction, evaluation and serving can all be combined. This can make use of the
 abstraction involved in the Tensorflow Estimator API.
I am not able to verify the performance of the model using Bidirectional LSTMs due to computational constraints.
