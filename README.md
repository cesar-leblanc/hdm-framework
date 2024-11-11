<a name="readme-top"></a>

<p align="center">
  <a href="https://github.com/cesar-leblanc/hdm-framework/graphs/contributors"><img src="https://img.shields.io/github/contributors/cesar-leblanc/hdm-framework" alt="GitHub contributors"></a>
  <a href="https://github.com/cesar-leblanc/hdm-framework/network/members"><img src="https://img.shields.io/github/forks/cesar-leblanc/hdm-framework" alt="GitHub forks"></a>
  <a href="https://github.com/cesar-leblanc/hdm-framework/issues"><img src="https://img.shields.io/github/issues/cesar-leblanc/hdm-framework" alt="GitHub issues"></a>
  <a href="https://github.com/cesar-leblanc/hdm-framework/blob/main/LICENSE"><img src="https://img.shields.io/github/license/cesar-leblanc/hdm-framework" alt="License"></a>
  <a href="https://github.com/cesar-leblanc/hdm-framework/pulls"><img src="https://img.shields.io/github/issues-pr/cesar-leblanc/hdm-framework" alt="GitHub pull requests"></a>
  <a href="https://github.com/cesar-leblanc/hdm-framework/stargazers"><img src="https://img.shields.io/github/stars/cesar-leblanc/hdm-framework" alt="GitHub stars"></a>
  <a href="https://github.com/cesar-leblanc/hdm-framework/watchers"><img src="https://img.shields.io/github/watchers/cesar-leblanc/hdm-framework" alt="GitHub watchers"></a>
</p>


<div align="center">
  <img src="Images/logo.png" alt="Project logo" width="100">
  <h2 align="center">hdm-framework</h2>
  <p align="center">A classification framework to enhance your habitat distribution models</p>
  <a href="https://github.com/cesar-leblanc/hdm-framework">View framework</a>
  ·
  <a href="https://github.com/cesar-leblanc/hdm-framework/issues">Report Bug</a>
  ·
  <a href="https://github.com/cesar-leblanc/hdm-framework/issues">Request Feature</a>
  <h1></h1>
</div>

This is the code for the framework of the paper [Phytosociology meets artificial intelligence: a deep learning classification framework for biodiversity monitoring of European flora through accurate habitat type prediction based on vegetation-plot records](https://arxiv.org/) published in Applied Vegetation Science.  
If you use this code for your work and wish to credit the authors, you can cite the paper:
```
@article{leblanc2023phytosociology,
  title =        {Phytosociology meets artificial intelligence: a deep learning classification framework for biodiversity monitoring of European flora through accurate habitat type prediction based on vegetation-plot records},
  author =       {Leblanc, César and Bonnet, Pierre and Servajean, Maximilien and Joly, Alexis, and others},
  journal =      {Applied Vegetation Science},
  volume =       {xxx},
  number =       {xxx},
  pages =        {xxx--xxx},
  year =         {2023},
  publisher =    {Wiley Online Library}
}
```
This framework aims to facilitate the training and sharing of Habitat Distribution Models (HDMs) using various types of input covariates including cover abundances of plant species and information about plot location.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Data](#data)
- [Installation](#installation)
- [Examples](#examples)
  - [Dataset](#dataset)
  - [Evaluation](#evaluation)
  - [Training](#training)
  - [Prediction](#prediction)
- [Models](#models)
- [Roadmap](#roadmap)
- [Unlicense](#unlicense)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Team](#team)
- [Structure](#structure)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prerequisites

Python version 3.7 or higher and CUDA are required.

On many systems Python comes pre-installed. You can try running the following command to check and see if a correct version is already installed:
```script
python --version
```
If Python is not already installed or if it is installed with version 3.6 or lower, you will need to install a functional version Python on your system by following the [official documentation](https://www.python.org/downloads/) that contains a detailed guide on how to setup Python.

To check whether CUDA is already installed or not on your system, you can try running the following command:
```script
nvcc --version
```
If it is not, make sure to follow the instructions [here](https://developer.nvidia.com/cuda-downloads).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data

The framework is optimized for data files from the European Vegetation Archive (EVA). These files contain all the information required for the proper functioning of the framework, i.e., for each vegetation plot the full list of vascular plant species, the estimates of cover abundance of each species, the location and the EUNIS classification. Once the database is downloaded (more information [here](http://euroveg.org/eva-database)), make sure you rename species and header data files respectively as `eva_species.csv` and `eva_header.csv`. All columns from the files are not needed, but if you decide to remove some of them to save space on your computer, make sure that the values are still tab-separated and that you keep at least:
- the columns `PlotObservationID`, `Matched concept` and `Cover %` from the species file (vegetation-plot data)
- the columns `PlotObservationID`, `Cover abundance scale`, `Date of recording`, `Expert System`, `Longitude` and `Latitude` from the header file (plot attributes)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation

Firstly, `hdm-framework` can be installed via repository cloning:
```script
git clone https://github.com/cesar-leblanc/hdm-framework.git
cd hdm-framework
```

Secondly, make sure that the dependencies listed in the `environment.yml` and `requirements.txt` files are installed.
One way to do so is to use `conda`:
```script
conda env create -f environment.yml
conda activate hdm-env
```

Thirdly, to check that the installation went well, use the following command:
```script
python main.py --pipeline 'check' 
```

If the framework was properly installed, it should output:
```script
No missing files.

No missing dependencies.

Environment is properly configured.
```

Make sure to place the species and header data files inside the `Datasets` folder before going further.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Examples

### Dataset

To pre-process the data from the European Vegetation Archive and create the input data and the target labels, run the following command:
```script
python main.py --pipeline 'dataset' 
```

Some changes can be made from this command to create another dataset. Here is an example to only keep vegetation plots from France and Germany who were recorded after 2000 and classified to the level 2 of the EUNIS hierarchy:
```script
python main.py --pipeline 'dataset' --countries 'France, Germany' --min_year 2000 --level 2
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Evaluation

To evaluate the parameters of a classifier on the dataset previously obtained using cross validation, run the following command:
```script
python main.py --pipeline 'evaluation' 
```

Some changes can be made from this command to evaluate other parameters. Here is an example to evaluate a TabNet Classifier using the top-3 macro average multiclass accuracy:
```script
python main.py --pipeline 'evaluation' --model 'tnc' --average 'macro' --k 3 
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Training

To train a classifier from the dataset previously obtained and save its weights, run the following command:

```script
python main.py --pipeline 'training' 
```

Some changes can be made from this command to train another classifier. Here is an example to train a Random Forest Classifier with 50 trees using the cross-entropy loss:
```script
python main.py --pipeline 'training' --model 'rfc' --n_estimators 50 -- criterion 'log_loss'
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Prediction

Before making predictions, make sure you include two new files that describe the vegetation data of your choice in the `Datasets` folder: `test_species.csv` and `test_header.csv`. The two files should contain the following columns (with tab-separated values):
- `PlotObservationID` (integer), `Matched concept` (string) and `Cover %` (float) for the species data, which respectively describe the plot identifier, the taxon names and the percentage cover
- `PlotObservationID` (integer), `Longitude` (float) and `Latitude` (float) for the header data, which respectively describe the plot identifier, the plot longitude and the plot latitude


To predict the classes of the new samples using a previously trained classifier, make sure the weights of the desired model are stored in the `Models` folder and then run the following command:
```script
python main.py --pipeline 'prediction' 
```

Some changes can be made from this command to predict differently. Here is an example to predict using a XGBoosting Classifier without the external criteria nor the GBIF normalization:

```script
python main.py --pipeline 'prediction' --model 'xgb' --features 'species' --gbif_normalization False
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Models

This section lists every major frameworks/libraries used to create the models included in the project:

* [![PyTorch](https://img.shields.io/badge/PyTorch-%23ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/) - MultiLayer Perceptron classifier (MLP)
* [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23f89a36.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org) - Random Forest Classifier (RFC)
* [![XGBoost](https://img.shields.io/badge/XGBoost-%23189fdd.svg?logo=read-the-docs&logoColor=white)](https://xgboost.readthedocs.io/) - XGBoost classifier (XGB)
* [![pytorch_tabnet](https://img.shields.io/badge/TabNet-%232f363d.svg?logo=github&logoColor=white)](https://github.com/dreamquark-ai/tabnet) - TabNet Classifier (TNC)
* [![RTDL](https://img.shields.io/badge/RTDL-%23ef5350.svg?logo=github&logoColor=white)](https://yura52.github.io/rtdl/stable/index.html) - Feature Tokenizer + Transformer classifier (FTT)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

This roadmap outlines the planned features and milestones for the project. Please note that the roadmap is subject to change and may be updated as the project progress.

- [ ] Implement multilingual user support
    - [x] English
    - [ ] French
- [ ] Integrate new popular algorithms
    - [x] MLP
    - [x] RFC
    - [x] XGB
    - [x] TNC
    - [x] FTT
    - [ ] KNN
    - [ ] GNB
- [ ] Add more habitat typologies
    - [x] EUNIS
    - [ ] NPMS
- [ ] Include other data aggregators
    - [x] EVA
    - [ ] TAVA
- [ ] Offer several powerful frameworks
    - [x] PyTorch
    - [ ] TensorFlow
    - [ ] JAX
- [ ] Allow data parallel training
    - [x] Multithreading
    - [ ] Multiprocessing
- [ ] Supply different classification strategies
    - [x] Top-k classification
    - [ ] Average-k classification

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Unlicense

This framework is distributed under the Unlicense, meaning that it is dedicated to public domain. See `UNLICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

If you plan to contribute new features, please first open an issue and discuss the feature with us. See `CONTRIBUTING.md` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Troubleshooting

It is strongly unadvised to:
- not perform normalization of species names against the GBIF backbone, as it could become a major obstacle in your ecological studies if you seek to combine multiple datasets
- not include the external criteria when preprocessing the datasets, as it could lead to inconsistencies while training models or making predictions

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Team

hdm-framework is a community-driven project with several skillful engineers and researchers contributing to it.  
hdm-framework is currently maintained by [César Leblanc](https://github.com/cesar-leblanc) with major contributions coming from [Alexis Joly](https://github.com/alexisjoly), [Pierre Bonnet](https://github.com/bonnetamap), [Maximilien Servajean](https://github.com/maximiliense), and the amazing people from the [Pl@ntNet Team](https://github.com/plantnet) in various forms and means.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Structure

    ┌── data                               <- Folder containing data-related scripts.
    │   ├── __init__.py                    <- Initialization script for the 'data' package.
    │   ├── load_data.py                   <- Module for loading data into the project.
    │   ├── preprocess_data.py             <- Module for data preprocessing operations.
    │   └── save_data.py                   <- Module for saving data or processed data.
    │
    ├── Data                               <- Folder containing the created data.
    ├── Datasets                           <- Folder containing various datasets for the project.
    │   ├── EVA                            <- Folder containing original EVA datasets.
    │   ├── NPMS                           <- Folder containing original NPMS datasets.
    │   ├── arborescent_species.npy        <- List of all arborescent species.
    │   ├── digital_elevation_model.tif    <- Digital elevation model data in TIFF format.
    │   ├── eunis_habitats.xlsx            <- Excel file containing the list of EUNIS habitat.
    │   ├── red_list_habitats.xlsx         <- Excel file containing the list of red list habitat data.
    │   ├── ecoregions.dbf                 <- Database file for ecoregion data.
    │   ├── ecoregions.prj                 <- Projection file for ecoregion shapefile.
    │   ├── ecoregions.shp                 <- Shapefile for ecoregion data.
    │   ├── ecoregions.shx                 <- Index file for ecoregion shapefile.
    │   ├── united_kingdom_regions.dbf     <- Database file for United Kingdom regions data.
    │   ├── united_kingdom_regions.prj     <- Projection file for United Kingdom regions shapefile.
    │   ├── united_kingdom_regions.shp     <- Shapefile for United Kingdom regions data.
    │   ├── united_kingdom_regions.shx     <- Index file for United Kingdom regions shapefile.
    │   ├── vegetation.dbf                 <- Database file for vegetation data.
    │   ├── vegetation.prj                 <- Projection file for vegetation shapefile.
    │   ├── vegetation.shp                 <- Shapefile for vegetation data.
    │   ├── vegetation.shx                 <- Index file for vegetation shapefile.
    │   ├── world_countries.dbf            <- Database file for world countries data.
    │   ├── world_countries.prj            <- Projection file for world countries shapefile.
    │   ├── world_countries.shp            <- Shapefile for world countries data.
    │   ├── world_countries.shx            <- Index file for world countries shapefile.
    │   ├── world_seas.dbf                 <- Database file for world seas data.
    │   ├── world_seas.prj                 <- Projection file for world seas shapefile.
    │   ├── world_seas.shp                 <- Shapefile for world seas data.
    │   └── world_seas.shx                 <- Python script (details needed).
    │
    ├── Experiments                        <- Folder for experiment-related files.
    │   ├── ESy                            <- Folder containing the expert system.
    │   ├── cmd_lines.txt                  <- Text file with command line instructions.
    │   ├── data_visualization.ipynb       <- Jupyter notebook for data visualization.
    │   ├── results_analysis.ipynb         <- Jupyter notebook for results analysis.
    │   ├── model_interpretability.py      <- Module for model interpretability.
    │   └── test_set.ipynb                 <- Jupyter notebook for creating a test set.
    │
    ├── Images                             <- Folder for image resources.
    │   ├── hdm-framework.pdf              <- Overview of hdm-framework image.
    │   ├── logo.png                       <- Project logo image.
    │   ├── neuron-based_models.pdf        <- Key aspect of neuron-based models image.
    │   ├── transformer-based_models.pdf   <- Key aspect of transformer-based models image.
    │   └── tree-based_models.pdf          <- Key aspect of tree-based models image.
    │
    ├── models                             <- Folder for machine learning models.
    │   ├── ftt.py                         <- Module for the FTT model.
    │   ├── __init__.py                    <- Initialization script for the 'models' package.
    │   ├── mlp.py                         <- Module for the MLP model.
    │   ├── rfc.py                         <- Module for the RFC model.
    │   ├── tnc.py                         <- Module for the TNC model.
    │   └── xgb.py                         <- Module for the XGB model.
    │
    ├── Models                             <- Folder containing the trained models.
    ├── pipelines                          <- Folder containing pipeline-related scripts.
    │   ├── check.py                       <- Module for checking the configuration.
    │   ├── dataset.py                     <- Module for creating the train dataset.
    │   ├── evaluation.py                  <- Module for evaluating the models.
    │   ├── __init__.py                    <- Initialization script for the 'pipelines' package.
    │   ├── prediction.py                  <- Module for making predictions.
    │   └── training.py                    <- Module for training the models.
    │
    ├── .github                            <- Folder for GitHub-related files.
    │   ├── ISSUE_TEMPLATE                 <- Folder for issues-related files.
    │   │   ├── bug_report.md              <- Template for reporting bugs.
    │   │   └── feature_request.md         <- Template for requesting new features.
    │   │
    │   └── pull_request_template.md       <- Template for creating pull requests.
    │
    ├── cli.py                             <- Command-line interface script for the project.
    ├── CODE_OF_CONDUCT.md                 <- Code of conduct document for project contributors.
    ├── CONTRIBUTING.md                    <- Guidelines for contributing to the project.
    ├── environment.yml                    <- YAML file specifying project dependencies.
    ├── __init__.py                        <- Initialization script for the root package.
    ├── main.py                            <- Main script for running the project.
    ├── README.md                          <- README file containing project documentation.
    ├── requirements.txt                   <- Text file listing project requirements.
    ├── SECURITY.md                        <- Security guidelines and information for the project.
    ├── UNLICENSE.txt                      <- License information for the project (Unlicense).
    └── utils.py                           <- Utility functions for the project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
