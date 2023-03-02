# Coding Challenge

The test.py file will perform the following operations
* Perform inference on a folder of example images
* Create and plot a confusion matrix
* Compute the Expected Calibration Error (ECE) and Max Calibration Error (MCE)
* Save false positives of each class in a subfolder of ‘results/false_positives’
* Generate the potential patterns figure of false positive
### Project Structure

    .
    ├── test.py                 #  Evaluate the performance (main file)
    ├── runner.py               #  Model related operation such as train, test, save, load the pretrained model.
    ├── models                  # 
    │   └── base_model.py       #  Build the Model Class
    ├── utils                   #  Useful functions
    │   ├── __init__.py         #  To mark directories on disk as Python package directories
    │   └── utils                 
    ├── results                 #  Store Experimental Result
    └── ...

## Usage of test.py file
$ python test.py --test_dir 'cifar10/test'

Arguments to perform the model performance evulation.
```
optional arguments:
  --test_dir         path to the test folder 
  --model            which model to use ['tiny', 'small', 'medium', 'large'] Different ConvNext Models
  --checkpoint       the path to load saved model default="./results/checkpoints/epoch_3.pth"
  --run_train        train the model
  --train_epochs     number of epochs to train
```


## Confusion Matrix on the Test Dataset of Cifar10

![alt text](results/confusion_matrix.png "Confusion Matrix")

## Calibration Graph on the Test Dataset of Cifar10

![alt text](results/calibrated_graph.png "Calibration Graph")

## Error Distributions among Different Classess
![alt text](results/error_ratio_car.png "False Positives of Car Class")

## Dependencies

The build-time and runtime dependencies of this repository are:

* [numpy](http://www.numpy.org/)
* [scikit-fmm](https://github.com/scikit-fmm)
* [scikit-image](https://github.com/scikit-image)
* [tqdm](https://github.com/noamraph/tqdm)
