# Breast Cancer Predictor
This is our Breast Cancer Predictor, it uses multiple attributes pertaining to Breast Cancer to predict 
whether or not someone may have a tumor and whether it is beneign or needs further investigation. It does this by utilizing 
a dataset of Breast Cancer Data and by utilizing different Classifiers.

## Running
To run the project, you will need to make sure you have a conda kernel and have pip install the two prerequisites listed:
<li> numpy
<li> tabulate
<li> matplotlib
  
Then launch the Final_Report.ipynb and run through all the cells to view the Report and at the bottom of the Report you will be able to run each classifier with your own test data

## Organization
+ Final_Report.ipynb - The Final Report and iteration of our Project
+ mid_demo.ipynb - The demo iteration of our project (outdated)
+ proposal.ipynb - The original project idea, this utilized a different dataset
+ mysklearn
  + classifiers.py - The classifiers used in out project
  + classifier_utils.py - The utility functions used in our project and for our classifiers
  + evaluators.py - The evaluators and evaluation functions used for the project
  + mypytable.py - Our custom table datatype used in our project
+ test_classifiers.py - Unit tests for our Classifiers
+ input_data
  + breastcancer.csv - Pre-cleaned data
+ output_data
  + breast_cancer_clean.csv - Cleaned data
