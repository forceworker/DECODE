# DECODE: Deep learning-based common deconvolution framework for various omics data

**DECODE** is a deep learning framework designed for solving deconvolution problems across various omics. It utilizes cell abundance as an intermediary to address the integration of multi-omics information at the tissue level. DECODE integrates contrastive learning, adversarial training, and other approaches into a computational framework, achieving the highest accuracy in deconvolution tasks across multiple scenarios.
<p align="center">
  <img width="60%" src="https://github.com/forceworker/DECODE/blob/main/fig/fig.png">
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


#### sc_data

Predefine the single-cell data (in h5ad format) used for mixing training and testing data, where the cell types are indicated by the CellType attribute in the obs property.Read the corresponding h5ad data using the anndata library:
```
import anndata as ad
train_data_file = 'data/lung_rna/296C_train.h5ad'
test_data_file = 'data/lung_rna/302C_test.h5ad'
train_data = ad.read_h5ad(train_data_file)
test_data = ad.read_h5ad(test_data_file)
```

#### Mix data

Define a class for data preprocessing called data_process in DECODE, and use the fit function to generate mixed data. The data_process class can be imported in the DECODE directory:
```
from data.data_process import data_process
# The type_list represents the cell types in the h5ad data, while train_sample_num indicates 
# the number of training data samples to be generated. The tissue_name refers to the name of 
# the tissue being analyzed, which is relevant for file naming. Similarly, test_sample_num specifies 
# the number of test data samples to be generated. Additionally, sample_size denotes the cell capacity of a single pseudo-tissue, 
# and num_artificial_cells indicates the number of artificial cells to be generated for simulating noise.
dp = data_process(type_list, train_sample_num=6000, tissue_name='lung_rna', 
                  test_sample_num=1000, sample_size=30, num_artificial_cells=30)
```

#### Train model

Initialize the stage2 and stage3 models using the DANN class and the MBdeconv class defined in DECODE, respectively. Call the train function within the classes to train the models.

```
from model.deconv_model_with_stage_2 import MBdeconv
from model.stage2 import *
# Initialize the stage2 module.
STAGE2 = DANN(epoches, batchsize, learning_rate)

# The train function receives the training dataset, as well as the test and validation datasets, 
# with the validation dataset being derived from the training dataset. The patience parameter 
# represents the patience value for the early stopping mechanism.
pred_loss, disc_loss, disc_loss_DA, best_model_weights = STAGE2.train(train_data, test_data, valid_data, patience = 3) 

# Initialize the stage3 module.
model = MBdeconv(num_feat, feat_map_w, feat_map_h, num_cell_type, epoches, Alpha, Beta, train_dataloader, test_dataloader)

# If training on a GPU is required, you need to set the GPU device.
device = torch.device('cuda')
if model.gpu_available:
    model = model.to(model.gpu)

# Transfer the encoder parameters from STAGE2 to STAGE3 and fix them.
STAGE2.encoder_da.load_state_dict(best_model_weights['encoder'])
encoder_params = copy.deepcopy(STAGE2.encoder_da.state_dict())
model.encoder.load_state_dict(encoder_params)

# Train stage 3.
loss1_list, loss2_list, nce_loss_list = model.train_model(model_save_name, True, patience)
```

#### Model Prediction
By using the predict function defined in DECODE, you can obtain the computed evaluation metrics such as CCC, RMSE, and Pearson's r, as well as the prediction results and the labels of the test dataset.

```
# The predict function can be used to obtain the prediction results 
# and evaluation metrics for the test data, as well as the model's predicted 
# results, denoted as pred.if_pure determines whether to enable the denoising mode.
CCC, RMSE, Corr, pred, gt = predict(test_dataloader, type_list, model_test, if_pure)
```

In Google Colab (https://colab.research.google.com/github/forceworker/DECODE/blob/main/train_lung_rna_colab.ipynb), there is a complete data example for Scenario 1, which allows you to fully run the DECODE method for learning purposes.

### Notation
The Jupyter records of the various experiments in the DECODE work can be found at: https://doi.org/10.5281/zenodo.15687743.


The cross-omics experiments and multi-omics feature-sharing experiments can be found at https://doi.org/10.5281/zenodo.15708922.