## Setup
Clone the repository and create the Conda environment:
```bash
git clone <REPO_URL>
cd Thesis_Project_Szabolcs_Pal
bash setup_env.sh
# if you open a new shell later:
conda activate ./my_local_env


## Data (LibriSpeech)
# Download the following from [OpenSLR #12](https://www.openslr.org/12):
- `train-clean-100.tar.gz`
- `train-clean-360.tar.gz`
- `dev-clean.tar.gz`

**Place the downloaded files in the `data/` folder** of this repository and extract them **there**. 
# Example (macOS):
```bash
# from repo root
mv ~/Downloads/train-clean-100.tar.gz data/
mv ~/Downloads/train-clean-360.tar.gz data/
mv ~/Downloads/dev-clean.tar.gz data/

#Train the models
#Run the ensemble (phonetic/spectral) pipeline, then the transformers pipeline:

#bash
PYTHONPATH=. python scripts/ensemble_operater/FINAL_PIPELINE_ENSEMBLE_MODELS.py
PYTHONPATH=. python scripts/transformers_operater/111_RUN_TRANSFORMERS_PIPELINE.py


## Repository layout

```text

scripts/                                # All runnable pipelines and utils
├─ ensemble_operater/                   # Runs ensemble (phonetic/spectral) pipelines
│  └─ __pycache__/                      
├─ evaluation/                          # Metrics, error analysis, extractor functions
│  └─ __pycache__/                      
├─ grid_build/                          # Builds enrollment "grid" configs (e.g., 10s_10i, 20s_20i)
├─ preprocessing/                       # Audio preprocessing + feature extraction                             
│  └─ __pycache__/                      
├─ training/                            # Training code for embedding/ensemble components
│  └─ __pycache__/                      
└─ transformers_operater/               # Pipelines for transformer-based models
   └─ __pycache__/                      

data/                                   # Put LibriSpeech archives/extracted folders here 

my_datasets/                            # Dataset generation code for speaker embedding models
└─ __pycache__/                         

models/                                 # Neural network model files
└─ __pycache__/                         



