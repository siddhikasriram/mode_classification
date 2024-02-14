# Laser Mode Modal Decomposition using Convolutional Neural Networks

This repository contains scripts and notebooks for Laser Mode Modal Decomposition using Convolutional Neural Networks (CNNs). The goal is to classify different laser modes based on low or high noise using generated dataset images and applying CNNs. 

## Tools and Technologies

- **Scripting Language:** Python 3.9
- **Computational Platform:** Google Colab with T4 GPU

## File Structure

- `lightpipes_data_generate.py`: Python script for generating dataset images.
- `classification_3_modes_low_noise.ipynb`: Jupyter notebook for classifying 3 laser modes with low noise.
- `classification_5_modes_low_noise.ipynb`: Jupyter notebook for classifying 5 laser modes with low noise.
- `classification_5_modes_high_noise.ipynb`: Jupyter notebook for classifying 5 laser modes with high noise.
- `dataset_folder_temp`: Folder containing generated dataset images.

## Instructions

1. **Installation:** Ensure you have the necessary dependencies installed by running `pip install LightPipes`.
2. **Setup:** Rename the `dataset_folder_temp` and execute `lightpipes_data_generate.py`, setting the `Ntot` variable to specify the number of images needed in the dataset.
3. **Dataset Handling:** Zip the generated dataset folder and add it to Google Drive.
4. **Execution:** Execute the provided Jupyter notebooks on Google Colab by adjusting the path names in the code.
5. **Output:** Run all cells in the notebooks to obtain the desired outputs.

## Additional Information

For detailed documentation and explanations, please refer to the respective code files in this repository.

## Author and Contact

- **Author:** Siddhika Sriram
- **Contact:** [Email](mailto:siddhikasriram.ss@gmail.com), [LinkedIn](https://www.linkedin.com/in/sidsriram/)
