# DECODE: Deep learning-based common deconvolution framework for various omics data

**DECODE** is a deep learning framework designed for solving deconvolution problems across various omics. It utilizes cell abundance as an intermediary to address the integration of multi-omics information at the tissue level. DECODE integrates contrastive learning, adversarial training, and other approaches into a computational framework, achieving the highest accuracy in deconvolution tasks across multiple scenarios.
<p align="center">
  <img width="60%" src="https://github.com/forceworker/DECODE_decov/blob/main/fig/fig.png">
</p>
More details can be found in paper.

## Setup

### Dependencies and Installation

Workflow of DECODE are implemented in python.The Python libraries used by DECODE and their specific versions are saved in the environment.yml.

Create a new environment using environment.yml to support running DECODE. The specific steps are as follows:

Step1:Type the directory where environment.yml is located in the terminal:

	> cd ~/DECODE  

Step2:Create the environment with a custom name:

	> conda env create --name env_name -f environment.yml  

Step3:Activate the environment:

	> conda activate env_name 

### Usage

The specific usage process can be referenced in the Jupyter notebooks of various experiments in DECODE.

#### sc_data

Predefine the single-cell data (in h5ad format) used for mixing training and testing data, where the cell types are indicated by the CellType attribute in the obs property.

#### Mix data

Define a class for data preprocessing called data_process in DECODE, and use the fit function to generate mixed data.

#### Train model

Initialize the stage2 and stage3 models using the DANN class and the MBdeconv class defined in DECODE, respectively. Call the train function within the classes to train the models.

#### Stage 4 (Model Prediction)
By using the predict function defined in DECODE, you can obtain the computed evaluation metrics such as CCC, RMSE, and Pearson's r, as well as the prediction results and the labels of the test dataset.


