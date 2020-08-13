Hello potential ML6 colleague!

The aim is to classify images.

## The data

Not included
## The model

In the trainer folder, you will be able to see several python files. The data.py, task.py and final_task.py files 

## Deploying the model


gcloud ml-engine local train --module-name trainer.file_name .py --package-path trainer/
```

```
MODEL_NAME=<your_model_name>
VERSION=<your_version_of_the_model>
gcloud ml-engine predict --model $MODEL_NAME --version $VERSION --json-instances check_deployed_model/test.json
```

