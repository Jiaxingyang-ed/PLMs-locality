# Locality Convergence Dynamics Reveal Irreducible Global Dependencies in Protein Language Model Representations

This repository contains the code and data for analyzing the locality of protein language model (PLM) representations. The project investigates how local sequence windows can predict global residue embeddings, revealing irreducible global dependencies that correlate with experimental conformational heterogeneity.

## Project Overview

The core research questions addressed in this project:

1. **Locality Analysis**: Can a "local encoder" trained on local sequence windows predict global residue embeddings?
2. **Convergence Dynamics**: How does prediction error converge as window size increases?
3. **Irreducible Residuals**: What is the residual plateau `c` that cannot be explained by local context?
4. **Experimental Correlation**: Does the irreducible residual correlate with conformational heterogeneity (NOC clusters)?
5. **Cross-Model Validation**: Are findings consistent across different PLMs (ProtT5 vs ESM-2)?

## Directory Structure

```
P53_Rescue/
├── config.py                    # All paths & constants
├── models.py                    # Shared model classes (LocalSeqEncoder, etc.)
├── locality_probing/            # Main experiments
│   ├── local_encoder/          # Training scripts & models
│   ├── p53_analysis/           # P53 convergence analysis
│   ├── taxonomy_validation/    # Taxonomy validation
│   ├── structure_analysis/     # Structural metrics
│   └── scale_validation/       # Scale validation
├── cross_model_esm2/            # ESM-2 cross-model validation
├── representation_analysis/     # Rotation test & probing
├── data/                        # Intermediate data (not committed)
│   ├── embeddings/             # Processed embeddings
│   ├── ppi/                    # PPI data
│   ├── skempi/                 # SKEMPI data
│   └── pdb/                    # PDB structure files
├── raw_data/                    # Original data
├── results/                     # Output figures/tables
├── legacy/                      # Archive of old/experimental code
├── requirements.txt
└── .gitignore
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- transformers (for ESM-2 models)
- numpy, pandas, scipy
- scikit-learn
- matplotlib, seaborn
- tqdm
- h5py

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd P53_Rescue
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Edit `config.py` to set paths to your data directories and model parameters.

### Training Local Encoders

**ProtT5 Local Encoder:**
```bash
cd locality_probing/local_encoder
python train_local_encoder.py
```

**ESM-2 Local Encoder:**
```bash
cd cross_model_esm2
python 02_train_esm2_local_encoder.py
```

### P53 Convergence Analysis

```bash
cd locality_probing/p53_analysis
python analyze_convergence.py
```

### Taxonomy Validation

```bash
cd locality_probing/taxonomy_validation
python validate_window_scales.py
```

### Cross-Model Validation

```bash
cd cross_model_esm2
python 03_analyze_p53_esm2.py
```

## Key Results

The main findings from this research:

1. **Convergence Plateau**: Prediction error converges to a non-zero plateau `c` as window size increases, indicating irreducible global dependencies.
2. **Correlation with Heterogeneity**: The residual plateau `c` correlates with experimental conformational heterogeneity (NOC clusters from NMR).
3. **Cross-Model Consistency**: Similar convergence patterns observed across ProtT5 and ESM-2 models.
4. **Taxonomy Validation**: Findings generalize across diverse protein families and taxonomic groups.

## Data Sources

- **STRING Database**: Protein-protein interaction networks (https://string-db.org/cgi/download.pl)
- **SKEMPI**: Database of binding affinity changes upon mutation (https://skempi2.ch.icm.uu.se/)
- **ProtT5-XL**: Protein language model embeddings (https://github.com/agemagician/ProtTrans)
- **ESM-2**: Protein language model embeddings (https://github.com/facebookresearch/esm)
- **PDB**: Protein structure data (https://www.rcsb.org/)
- **NMR NOC Clusters**: Experimental conformational heterogeneity data

## Execution Order

To reproduce the results from raw data to final figures:

1. **Download raw data** (place in `raw_data/`):
   - STRING PPI data: `9606.protein.links.v12.0.txt`
   - ProtT5 embeddings: `9606.protein.sequence.embeddings.v12.0.h5`
   - SKEMPI data: `skempi_v2.xlsx`
   - P53 sequence: `p53_sequence.fasta`

2. **Process embeddings** (generate intermediate data in `data/`):
   - Extract residue embeddings from PLM outputs
   - Generate mutation datasets with local sequence windows
   - Compute global embedding deltas (Δh)

3. **Train local encoders**:
   ```bash
   # ProtT5 encoder
   python locality_probing/local_encoder/train_local_encoder.py
   
   # ESM-2 encoder
   python cross_model_esm2/02_train_esm2_local_encoder.py
   ```

4. **Analyze convergence dynamics**:
   ```bash
   # P53 three-metric analysis
   python locality_probing/p53_analysis/analyze_convergence.py
   
   # ESM-2 multi-metric analysis
   python cross_model_esm2/03_analyze_p53_esm2.py
   ```

5. **Validate across taxonomic groups**:
   ```bash
   python locality_probing/taxonomy_validation/validate_window_scales.py
   ```

6. **Generate final figures** (saved to `results/`):
   - Convergence curves for different metrics
   - Correlation plots with NOC clusters
   - Cross-model comparison figures

## Citation

If you use this code or data in your research, please cite:

```
[Add citation information when published]
```

## License

[Specify license - e.g., MIT, BSD, etc.]

## Contact

[Add contact information]

## Acknowledgments

- ProtT5-XL model from the Rost lab
- ESM-2 model from Meta AI
- STRING database consortium
- SKEMPI database maintainers
# PLMs-locality
