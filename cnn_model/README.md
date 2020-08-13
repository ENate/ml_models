Hello potential ML6 colleague!

If you are reading this, you are probably applying for an ML engineering job at ML6. This test will evaluate if you have the right skills for this job. The test should approximately take 2 hours.

In the test, you will try to classify the mugs we drink from at ML6. If you are able to complete this test in a decent way, you might soon be drinking coffee from the black ML6 mug (which is also in the data) together with us.

## The data

As you can see, all data can be found in the data folder. For your purposes, the data has already been split in training data and test data. They are respectively in the train folder and test folder. In those folders, you can find four folders which represent the mugs you'll need to classify. There are four kind of mugs: the white mug, the black mug (the ML6 mug), the blue mug and the transparent mug (the glass). The white mug is class 0, the black mug class 1, the blue mug class 2 and the transparent mug class 3. These class numbers are necessary to create a correct classifier. If you want, you can inspect the data, however, the code to load the data of the images into numpy arrays is already written for you.

## The model

In the trainer folder, you will be able to see several python files. The data.py, task.py and final_task.py files are already coded for you. The only file that needs additional code is the model.py file. The comments in this file will indicate which code has to be written.

To test how your model is doing you can execute the following command (you will need to [install](https://cloud.google.com/sdk/docs/) the gcloud command):

```
gcloud ml-engine local train --module-name trainer.task --package-path trainer/ -- --eval-steps 5
```

If you run this command before you wrote any code in the model.py file, you will notice that it returns errors. Your goal is to write code that does not return errors and achieves an accuracy that is as high as possible.

The command above will perform 5 evaluation steps during the training. If you want to change this, you only have to change the 5 at the end of the command to the number of evaluation steps you like. The batch size and the number of training steps should be defined in the model.py file.

Make sure you'll think about the solution you will submit for this coding test. If you want the code written by us can be changed to your needs. It is however important that we can still perform our automated evaluation when you submit your solution so make sure you test your solution thoroughly before you submit it. How you can test your solution will be explained later in this README.md file.

![Data overview](data.png =1x)

The command above uses the task.py file. As you can see in the figure above, this file only uses the mug images in the training folder of this repository and uses the test folder to evaluate the model. This is excellent to test how the model performs but to obtain a better evaluation one can also train upon all available data which should increase the performance on the dataset you will be evaluated. After you finished coding up model.py, you can read on and you'll notice how to train your model on the full dataset.

## Deploying the model

Once you've got the code working you will need to deploy the model to Google Cloud to turn it into an API that can receive new images of mugs and returns its prediction for this mug. Don't worry, the code for this is already written in the final_task.py file. To deploy the model you've just written, you only have to run a few commands in your command line.

To export your trained model and to train your model on the training folder and the test folder you have to execute the following command (only do this once you've completed coding the model.py file):

```
gcloud ml-engine local train --module-name trainer.final_task --package-path trainer/
```

Once you've executed this command, you will notice that the output folder was created in the root directory of this repository. This folder contains your saved model that you'll be able to deploy on Google Cloud ML-engine.

To be able to deploy the model on a Google Cloud ML-engine you will need to create a [Google Cloud account](https://cloud.google.com/). You will need a credit card for this, but you'll get free credit from Google to run your ML-engine instance.

Once you've created your Google Cloud account, you'll need to deploy your model on a project you've created. You can follow a [Google guideline](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction) for this.

## Checking your deployed model

Before you submit your test, you can check if your deployed model works the way it should by executing the following commands:

```
MODEL_NAME=<your_model_name>
VERSION=<your_version_of_the_model>
gcloud ml-engine predict --model $MODEL_NAME --version $VERSION --json-instances check_deployed_model/test.json
```

Check if you are able to get a prediction out of the gcloud command. If you get errors, you should try to resolve them before submitting the project. The output of the command should look something like this (the numbers will probably be different):

```
CLASSES  PROBABILITIES
1        [2.0589146706995187e-12, 1.0, 1.7370329621294728e-13, 1.2870057122347237e-32]
```

The values you use for the $MODEL_NAME variable and the $VERSION variable can be found in your project on the Google Cloud web interface. You will need these values and your Google Cloud project id to submit your coding test.

To be able to pass the coding test. You should be able to get an accuracy of 75% on our secret dataset of mugs (which you don't have access to). If your accuracy however seems to be less than 75% after we evaluated it, you can just keep submitting solutions until you are able to get an accuracy of 75%.

### Submitting your coding test

Once you are able to execute the command above without errors, you can add us to your project:

* Go to the menu of your project
* Click IAM & admin
* Click Add
* Add automated-evaluation@billing-skyhaus.iam.gserviceaccount.com with the Project Owner role

If you added us to your project you should fill in [this form](https://docs.google.com/forms/d/1A6LgwK6zoZVZG3vkDE823jpSc1Cw6VQ4aTd_07ILqwI) so we are able to automatically evaluate your test. Once you've filled in the form you should receive an email with the results of your coding test within 2 hours. We'll hope with you that your results are good enough to land an interview at ML6. If however you don't, you can resubmit a new coding test solution as many times you want so don't give up!

If you are invited for an interview at ML6 afterwards, you'll have to make sure that you bring your laptop with the code that you've wrote on it, so you can explain your model.py file to us.
