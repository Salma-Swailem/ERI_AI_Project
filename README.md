# Image Classification Project: Comparing SML, USML, and DL Approaches

## Overview

This project involves applying three different AI approaches—Supervised Machine Learning (SML), Unsupervised Machine Learning (USML), and Deep Learning (DL)—to the same image dataset and comparing their performance. The dataset consists of images of cats and dogs, and the goal is to classify these images accurately using various techniques.

## Dataset

The dataset is divided into training and testing sets:

-   **Training Set**: Contains images of cats and dogs used to train the models.
-   **Testing Set**: Contains images used to evaluate the models' performance.

# Directory Structure

```
main.py
project_directory/
│
├── training_set/
│   ├── dogs/
│   │   ├── dog1.jpg
│   │   ├── dog2.jpg
│   │   └── ...
│   └── cats/
│       ├── cat1.jpg
│       ├── cat2.jpg
│       └── ...
│
└── test_set/
    ├── dogs/
    │   ├── dog1.jpg
    │   ├── dog2.jpg
    │   └── ...
    └── cats/
        ├── cat1.jpg
        ├── cat2.jpg
        └── ...
```

## Requirements

-   Python 3.x
-   NumPy
-   OpenCV
-   scikit-learn
-   TensorFlow/Keras
-   Matplotlib

```bash
pip install numpy opencv-python scikit-learn tensorflow matplotlib
```

## Project Structure

-   `SML_code.py`: Contains the Supervised Machine Learning implementation using K-Nearest Neighbors (KNN) and Decision Tree classifiers.
-   `USML_code.py`: Contains the Unsupervised Machine Learning implementation using KMeans clustering.
-   `DL_code.py`: Contains the Deep Learning implementation using Convolutional Neural Networks (CNN).

## Installation

1.  Clone the repository.
    
    bash
    
    Copy code
    
    `git clone https://github.com/your-username/project-name.git` 
    
2.  Change to the project directory.
    
    bash
    
    Copy code
    
    `cd project-directory` 
    
3.  Create and activate a virtual environment.
    
    bash
    
    Copy code
    
    `python3 -m venv venv
    source venv/bin/activate` 
    
4.  Install the required packages.
    
    bash
    
    Copy code
    
    `pip install -r requirements.txt`

## How to Run

1.  Ensure the directory structure matches the one specified above.
2.  Place your dog and cat images in the `training_set` and `test_set` directories as shown.
3.  Run the `main.py` script

```bash
python main.py
```

## Code

### 1. Supervised Machine Learning (SML)
 [SML](https://github.com/Salma-Swailem/ERI_AI_Project/blob/01c8fa1d6001786dcd741079039e5f9a52cdf318/SML/temp.py)


### 2. Unsupervised Machine Learning (USML)
 [USML](https://github.com/Salma-Swailem/ERI_AI_Project/blob/01c8fa1d6001786dcd741079039e5f9a52cdf318/USML/scripts/main.py)

![Classfication KMeans and Ground Truth](USML/Elbow method for optimal K.png](https://github.com/Salma-Swailem/ERI_AI_Project/blob/main/USML/Classfication%20KMeans%20and%20Ground%20Truth.png)
![Elbow method for optimal K](https://github.com/Salma-Swailem/ERI_AI_Project/blob/main/USML/Elbow%20method%20for%20optimal%20K.png)
![KMeans Clustering Results (Test Data)](USML/KMeans Clustering Results (Test Data).png](https://github.com/Salma-Swailem/ERI_AI_Project/blob/main/USML/KMeans%20Clustering%20Results%20(Test%20Data).png)
![KMeans Clustering Results (Trainging Data)](https://github.com/Salma-Swailem/ERI_AI_Project/blob/main/USML/KMeans%20Clustering%20Results%20(Trainging%20Data).png)



### 3. Deep Learning (DL)
 [DL](https://github.com/Salma-Swailem/ERI_AI_Project/blob/01c8fa1d6001786dcd741079039e5f9a52cdf318/DL/DeepLearning_Project.py)



## Comparison and Results

After running all three approaches, compare the results based on accuracy, confusion matrices, and classification reports. Visualizations are also provided to illustrate the clustering results.
