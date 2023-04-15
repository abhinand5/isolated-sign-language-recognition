# Google - Isolated Sign Language Recognition
The goal of this competition is to classify isolated American Sign Language (ASL) signs.

Competition Link: https://www.kaggle.com/competitions/asl-signs/

## Getting started

Install Poetry

`$ pip install poetry`

Install the dependencies

`$ poetry install`

usage: main.py [-h] {train,eval,tflite-convert} ...

```bash
$ python main.py --help
ISLR CLI

positional arguments:
{train,eval,tflite-convert}
                        Mode of operation - [train, eval, inference]

optional arguments:
-h, --help            show this help message and exit
```

## Dataset Description
Deaf children are often born to hearing parents who do not know sign language. Your challenge in this competition is to help identify signs made in processed videos, which will support the development of mobile apps to help teach parents sign language so they can communicate with their Deaf children.

This competition requires submissions to be made in the form of TensorFlow Lite models. You are welcome to train your model using the framework of your choice as long as you convert the model checkpoint into the tflite format prior to submission. Please see the evaluation page for details.

## Models
TODO

## License
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
