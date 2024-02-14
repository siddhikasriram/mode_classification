# mode_classification

Laser Mode Modal Decomposition using Convolutional Neural Networks
Overview:
This repository contains code and data files for a Master's studies final project in the COMP 594 course. The project, conducted by Siddhika Sriram, focuses on using Convolutional Neural Networks (CNN) to decompose the modes of a laser system.

Tools and Technologies:
Scripting Language: Python 3.9
Computational Platform: Google Collab with T4 GPU
File Structure:
The repository includes:

lightpipes_data_generate.py: Python script for generating dataset images.
classification_3_modes_low_noise.ipynb: Jupyter notebook for classifying 3 laser modes with low noise.
classification_5_modes_low_noise.ipynb: Jupyter notebook for classifying 5 laser modes with low noise.
classification_5_modes_high_noise.ipynb: Jupyter notebook for classifying 5 laser modes with high noise.
dataset_folder_temp: Folder containing generated dataset images.
Instructions:
Installation: Ensure you have the necessary dependencies installed by running pip install LightPipes.
Setup: Rename the dataset_folder_temp and execute lightpipes_data_generate.py, setting the Ntot variable to specify the number of images needed in the dataset.
Dataset Handling: Zip the generated dataset folder and add it to Google Drive.
Execution: Execute the provided Jupyter notebooks on Google Collab by adjusting the path names in the code.
Output: Run all cells in the notebooks to obtain the desired outputs.
Additional Information:
For detailed documentation and explanations, please refer to the respective code files in this repository.

Author and Contact:
Siddhika Sriram - Please reach out for any further information or assistance.
