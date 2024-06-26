

# Capture of Causal Genetic Mechanisms by Machine Learning Methods

## Overview
This project evaluates the effectiveness of machine learning (ML) algorithms in identifying causal genetic mechanisms that influence bacterial traits. Using real genomic sequencing data from *Klebsiella pneumoniae*, we simulate quantitative phenotypes by varying the number of causal loci and the sizes of their effects. Our analysis includes four popular ML models: Elastic Nets, Random Forest, XGBoost, and Neural Networks.

## Installation

### Prerequisites
Ensure you have Python 3.x installed on your machine along with pip for package management. It is recommended to use a virtual environment for Python projects.

1. Clone the Repository

```bash
git clone https://github.com/gzhoubioinf/Simulation_and_Causal_ML.git
cd Simulation_and_Causal_ML
```

2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Data Preparation
Utilize the provided scripts to prepare your dataset for analysis:

Generate PLINK Data Files:

```bash
python generate_tfam_tped_data.py

```
Processes genomic data to generate unitigs.tped and unitigs.tfam for use with PLINK.

Run GWAS Simulations:
```bash
bash gwas_simu.sh
```
Performs genetic simulations using PLINK and GCTA, outputting results to the pheno data directory.

Generate SNP Lists:

```bash
python generate_snplist.py
```
Creates .snplist files based on specified sample sizes for further analysis.

Running Machine Learning Models
Execute the main script and specific model scripts:
```bash
python main.py
python main_ent.py  # For Elastic Net
python main_nn.py   # For Neural Network
python main_rf.py   # For Random Forest
python main_xgb.py  # For XGBoost

```

```bash
Example Data Files
sim_data.Rtab: Example genomic data.
out10.phen: Simulated phenotypic data output.
```



This project is licensed under the MIT License - see the LICENSE file for details.

---

