# Coding Challenge

## Usage of test.py file
$ python test.py --test_dir 'cifar10/test'

Arguments to perform the Rivulet2 tracing algorithm.

optional arguments:
  --model            which model to use ['tiny', 'small', 'medium', 'large']
  --checkpoint       the path to load saved model default="./results/checkpoints/epoch_3.pth"
  -o OUT, --out OUT     The name of the output file

The test.py file will perform the following operations
* Perform inference on a folder of example images
* Create and plot a confusion matrix
* Compute the Expected Calibration Error (ECE) and Max Calibration Error (MCE)
* Save false positives of each class in a subfolder of ‘results/false_positives’
* Generate the potential patterns figure of false positive

## Example Confusion Matrix

![alt text](results/confusion_matrix.png "Confusion Matrix")

## Example Calibration Graph

![alt text]((results/calibration_graph.png "Calibration Graph")

## Dependencies

The build-time and runtime dependencies of this repository are:

* [numpy](http://www.numpy.org/)
* [scikit-fmm](https://github.com/scikit-fmm)
* [scikit-image](https://github.com/scikit-image)
* [tqdm](https://github.com/noamraph/tqdm)
