# Google - Isolated Sign Language Recognition
The goal of this competition is to classify isolated American Sign Language (ASL) signs.

Competition Link: https://www.kaggle.com/competitions/asl-signs/

## Dataset Description
Deaf children are often born to hearing parents who do not know sign language. Your challenge in this competition is to help identify signs made in processed videos, which will support the development of mobile apps to help teach parents sign language so they can communicate with their Deaf children.

This competition requires submissions to be made in the form of TensorFlow Lite models. You are welcome to train your model using the framework of your choice as long as you convert the model checkpoint into the tflite format prior to submission. Please see the evaluation page for details.

## Getting started

Install Poetry

`$ pip install poetry`

Install the dependencies

`$ poetry install`

## Using the CLI:

This is the command-line interface (CLI) for the ISLR project. The CLI provides several sub-commands to perform different operations related to the project.

```bash
$ python main.py --help
usage: main.py [-h] {datagen,train,distill,eval,tflite-convert} ...

ISLR CLI

positional arguments:
  {datagen,train,distill,eval,tflite-convert}
                        Mode of operation - [train, distill, eval, tflite-convert]

optional arguments:
  -h, --help            show this help message and exit
```

### Configuration files
The CLI uses the following configuration files stores in the [conf folder](./conf/):

`train_config.yml`: Contains the configurations for training the model.

`eval_config.yml`: Contains the configurations for evaluating the model.

`inference_config.yml`: Contains the configurations for performing inference on the trained model.

### datagen

This sub-command preprocess and normalizes the raw data (also generated augmented variants to increase the dataset size) based on the configurations provided in the train_config.yml file. The generated data can be saved to the location specified in the config file using the `--save` flag. The resultant file is a `.npy` file.

**Usage:** `python main.py datagen [--save]`

**Arguments**:

`--save`: Setting this flag will save the data in the location set in the config file.

### train
This sub-command trains a model based on the configurations provided in the train_config.yml file. The `--dry-run` flag can be used to stop just before the model.fit() method is called. The `--save-feature-stats` flag can be used to save feature statistics used for normalization and preprocessing in a pickle dump.

**Usage:** `python main.py train [--dry-run] [--save-feature-stats]`

**Arguments**:

`--dry-run`: Dry run exits just before the model.fit() method is called.

`--save-feature-stats`: Saves the feature stats (used for normalization and preprocessing) in a pickle dump.

### distill

This sub-command performs knowledge distillation training based on the configurations provided in the train_config.yml file.

**Usage:** `python main.py distill`

### eval

This sub-command evaluates the performance of a trained model on the specified out-of-fold (OOF) fold using the weights of the model specified by the `--weights-path` flag.

**Usage:** `python main.py eval --fold-num <fold_num> --weights-path <weights_path>`

**Arguments**:

--fold-num: The OOF fold on which to evaluate the model (required).
--weights-path: The weights (*.h5) of the model to load and evaluate (required).

### tflite-convert
This sub-command converts a trained Keras model to TensorFlow Lite format. The input model is specified using the `--input flag`. The converted model is saved in the specified `--dest-dir` or the default model directory specified in the config file. The `--quantize` flag can be used to quantize the model during conversion. The `--quantize-method` flag can be used to specify the quantization method, which can be either "dynamic" or "float16".

Usage: `python main.py tflite-convert --input <input_path> [--dest-dir <dest_dir>] [--quantize] [--quantize-method <method>]`

**Arguments:**

`--input`: Path of the Keras model which has to be converted (*.h5) (required).

`--dest-dir`: Destination directory where the converted tflite model has to be saved (optional).

`--quantize`: Whether or not to quantize the model as part of conversion (optional, default False).

`--quantize-method`: Quantization method, either "dynamic" (default) or "float16" (optional).

### inference

This sub-command performs inference on a trained model using the configurations provided in the inference_config.yml file.

**Usage:** `python main.py inference`

## Models

[Will update soon]

## License
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
