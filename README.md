# CILP Assessment: Multimodal Learning

## Data Setup

The dataset consists of cubes and spheres each having an rgb image and matching data from a LiDAR sensor. From the original dataset ([kaggle](https://www.kaggle.com/datasets/andandand/cubes-and-spheres-lidar-and-rgb)) consisting of 9999 cubes and 9999 spheres I created a subset of 1000 cubes and 1000 spheres and uploaded it as a grouped fiftyone dataset on huggingface. All notebooks will start by loading this huggingface dataset. In the first notebook I explain how I created this subset and how to use it.

https://huggingface.co/datasets/MatthiasCr/multimodal-shapes-subset

## W&B Project

All training in this repo is logged to this public W&B project:

https://wandb.ai/matthiascr-hpi-team/cilp-extended-assessment/overview

![](results/wandb-dashboard.png)

## Running on Google Colab

The notebooks can be executed in google colab. 

- Open a notebook in colab by navigating to it on github and then change the url from github.com to githubtocolab.com.

- Setting a `HF_TOKEN=<your_huggingface_token>` as Colab Secret so the notbook can load data from huggingface. 

- Execute this cell which is the first in every notebook:

```python
import sys

# Colab-only setup
if "google.colab" in sys.modules:
    print("Running in Google Colab. Setting up repo")

    !git clone https://github.com/MatthiasCr/Computer-Vision-Assignment-2.git
    %cd Computer-Vision-Assignment-2
    !pip install -r requirements.txt
```

This clones the repo and sets it as working directory. Now local imports from the src module will work. It also installs all necessary python packages.

**W&B Setup** (only for notebooks 2-4)

- Open the file /content/Computer-Vision-Assignment-2/src/training.py and set WANDB_TEAM_NAME to your W&B team name and WANDB_PROJECT_NAME to a name for a (new) project

- Execute the cell with `!wandb login` and insert your W&B token