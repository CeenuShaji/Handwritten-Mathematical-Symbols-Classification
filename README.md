# Final Project

This is a **group assignment**.

## Code Implementation & Technical Report

The final deliverables include a 4-page IEEE-format report, code implementation and a detailed GitHub readme file.

The final project is due Tuesday, December 6 @ 11:59 PM. Find the complete [rubric](https://ufl.instructure.com/courses/455013/assignments/5244219) in the Canvas assignment.

## Training Data

The training data set is the same for every team in this course.

You can download the training data from the Canvas page:

* ["data_train.npy"](https://ufl.instructure.com/files/72247539/download?download_frd=1)
* ["t_train.npy"](https://ufl.instructure.com/files/72245951/download?download_frd=1)

## Edit this READ-ME

Please edit this read-me file with information about your project. You can find a [READ-ME template here](https://github.com/catiaspsilva/README-template).

## About The Project

This is a final group project for Fundamentals of Machine Learning class in which we were tasked with developing a machine learning system to classify handwritten mathematical symbols. The dataset used for this project was collected by each student enrolled in this course and contains a variety of 10 symbols. Our group, team square root implemented YOLOv5 for classification of the symbols and object detection. In this, we were able to train the model and make predictions of the test with an accuracy score above 90%. 

## Getting Started
Below are instructions and the different libraries and dependencies that are needed to recreate this project locally. 


### Dependencies

- PyTorch 1.13.0

```
conda install pytorch -c pytorch
```

- Pandas 1.5.2
```
pip install pandas 
or
pip3 install pandas
```
- NumPy 1.23.5
```
pip install numpy
```
- Pillow 9.3.0
```
pip install pillow
```
- Matplotlib 3.6.2
```
pip install matplotlib
```
- Scikit-learn 1.1.3
```
pip install -U scikit-learn
```
- Scikit-image 0.19.3
```
pip install scikit-image
```

- List of YOLOv5 [dependencies](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)

### Installation and Running the Code
1. Clone the final project repository
```
git clone https://github.com/UF-EEL5840-F22/final-project---code-report-team-square-root.git
```
2. Clone the YOLOv5 GitHub repository
```
git clone https://github.com/ultralytics/yolov5
```
3. Run train.ipynb to create the folder structure of training and validation data with corresponding bounding boxes, and generated NumPy arrays of testing data and its corresponding labels. Before doing so, specify in the Jupyter Notebook the preceeding path and folder name variables. By default it is set to output within /blue/eel5840/justin.rossiter/team_square_root
4. If you changed the path in the previous step, change the path in eel5840.yaml to account for this.
5. Move gpu_job_train.sh to within the yolov5/ GitHub repository directory. Move eel5840.yaml within the yolov5/data subfolder.
6. Activate your environment in HiperGator and submit the SLURM script.
7. When training is finished, the model can be found within yolov5/runs/train/exp#/weights, where # is the run number. Copy the trained model to the final project repository, or set the path within test.ipynb to account for this.
8. Run test.ipynb to evaluate the model on the generated test NumPy array.
## Usage

## Roadmap

## Contributing

## License

## Authors
Atayliya Irving

Catalina Murray

Justin Rossiter

Ceenu Shaji

## Acknowledgements
- Catia Silva
- University of Florida

## Thank You
