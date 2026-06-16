# PLMs-locality

## Investigating irreducible global dependency in Protein Language Model representations under mutational perturbations

This project introduces a locality-probing framework to quantify irreducible global dependency in Protein Language Model representations under mutational perturbations.

---

## Framework Overview

<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCALITY PROBING FRAMEWORK                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Global PLM Embedding (h_global)                                │
│           │                                                    │
│           │ Mutation Δ                                          │
│           ▼                                                    │
│  Mutated Embedding (h_mut)                                      │
│           │                                                    │
│           │ Δh = h_mut - h_global                               │
│           ▼                                                    │
│  Representation Shift (Δh)                                      │
│           │                                                    │
│           ├─────────────────────────────────────────────────┐  │
│           │                                                 │  │
│           ▼                                                 │  │
│  Local Encoder (trained on windows)                         │  │
│           │                                                 │  │
│           │ Predicted Δh_local                               │  │
│           ▼                                                 │  │
│  Local Prediction (Δh_local)                                 │  │
│           │                                                 │  │
│           │                                                 │  │
│  Geometric Decomposition:                                     │  │
│  • L_θ = 1 - cos(Δh_local, Δh_global)  (angular)            │  │
│  • L_m = |log(||Δh_local|| / ||Δh_global||)|  (magnitude)   │  │
│  • L_r = ||Δh_local - Δh_global|| / ||Δh_global||  (residual)│  │
│           │                                                 │  │
│           ▼                                                 │  │
│  Convergence Analysis across window radii r                   │  │
│           │                                                 │  │
│           ▼                                                 │  │
│  Irreducible Residual Plateau c                             │  │
│           │                                                 │  │
│           ▼                                                 │  │
│  Biological Correlation (NOC clusters, ΔΔG)                  │  │
│                                                           │  │
└─────────────────────────────────────────────────────────────┘  │
                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

</div>

---

## Key Findings

- **Directional irrecoverability**: Angular information (L_θ) remains irreducible even at large window radii in both ProtT5 and ESM-2
- **Mutation-specific residual plateaus**: The convergence plateau c exhibits mutation-specific behavior, not a universal constant
- **Convergence dynamics**: Prediction error follows exponential decay L(r) = A·exp(-α·r) + c with non-zero plateau c
- **Biological correlation**: The residual plateau c shows positive correlation with conformational heterogeneity (NOC clusters from NMR)
- **Orthogonality to stability**: c is orthogonal to thermodynamic stability (ΔΔG), capturing distinct biophysical properties
- **Cross-model differences**: Convergence geometry is architecture-dependent, with ProtT5 and ESM-2 showing different angular recovery patterns

---

## Scientific Motivation

Protein Language Models (PLMs) have revolutionized protein representation learning by capturing complex sequence-structure relationships. However, a fundamental question remains: **to what extent are PLM representations local or global?**

This matters because:

1. **Interpretability**: If representations are irreducibly global, local interpretability methods have fundamental limits
2. **Transfer learning**: Understanding locality boundaries informs effective transfer learning strategies
3. **Biological insight**: Global dependencies may reflect long-range allosteric mechanisms in proteins
4. **Model design**: Locality analysis guides architectural choices for protein sequence models

Traditional local-window approaches assume that local context can approximate global representations. Our framework tests this assumption by quantifying how much of a mutation-induced representation shift can be recovered from progressively larger sequence windows.

---

## Framework Overview

### Local Encoder Architecture

We train transformer-based local encoders to predict mutation-induced embedding shifts (Δh) from local sequence windows:

```
Δh_local = Encoder(local_sequence_window)
```

The encoder is trained on mutations from diverse proteins, learning to approximate global embedding changes from local context.

### Geometric Decomposition

We decompose prediction error into three orthogonal components:

- **Angular deviation (L_θ)**: Directional mismatch between predicted and actual shifts
  ```
  L_θ = 1 - cos(Δh_local, Δh_global)
  ```
  Measures whether the encoder captures the direction of representation change.

- **Magnitude deviation (L_m)**: Scale difference between predicted and actual shifts
  ```
  L_m = |log(||Δh_local|| / ||Δh_global||)|
  ```
  Measures whether the encoder captures the magnitude of representation change.

- **Relative residual (L_r)**: Overall normalized error
  ```
  L_r = ||Δh_local - Δh_global|| / ||Δh_global||
  ```
  Combines both directional and magnitude information.

### Convergence Analysis

We compute these metrics across increasing window radii r = {2, 5, 10, 15, 20, 30} residues and fit exponential decay:

```
L(r) = A·exp(-α·r) + c
```

where:
- A: initial error amplitude
- α: convergence rate
- c: irreducible residual plateau (cannot be eliminated by expanding context)

---

## Main Results

### Directional Irrecoverability

**Setup**: Trained local encoders on ProtT5-XL (1024-dim) and ESM-2 (640-dim) embeddings using mutations from diverse proteins.

**Finding**: Angular error L_θ converges to a non-zero plateau c_θ > 0 in both models, indicating that directional information is irreducibly global. Even at r=30 residues, local windows fail to capture the direction of representation change.

**Interpretation**: PLM representations encode long-range dependencies that cannot be approximated by local sequence context, suggesting fundamental architectural constraints on local interpretability.

**Corresponding figure**: Convergence curves for L_θ, L_m, L_r across radii (Figure 2 in manuscript)

### Convergence Dynamics and Plateau c

**Setup**: Fitted exponential decay to L_r across radii for individual mutations, extracting mutation-specific plateau c.

**Finding**: The plateau c varies significantly across mutations (c ∈ [0.1, 0.8]), indicating mutation-specific irreducible residuals rather than a universal constant.

**Interpretation**: Different mutations engage different degrees of global dependency, possibly reflecting mutation-specific allosteric networks or structural contexts.

**Corresponding figure**: Distribution of c across mutations (Figure 3 in manuscript)

### Biological Correlations

**Setup**: Correlated c with experimental conformational heterogeneity (NOC clusters from NMR) and thermodynamic stability (ΔΔG from SKEMPI).

**Finding**: c shows positive correlation with NOC clusters (ρ ≈ 0.6, p < 0.05) but no significant correlation with ΔΔG.

**Interpretation**: The irreducible residual captures conformational heterogeneity rather than stability, suggesting that PLMs encode dynamic ensemble properties.

**Corresponding figure**: Scatter plots of c vs NOC and c vs ΔΔG (Figure 4 in manuscript)

### Cross-Model Comparison

**Setup**: Compared convergence geometry between ProtT5-XL and ESM-2 on the same mutation set.

**Finding**: Both models show directional irrecoverability, but with different angular recovery patterns. ESM-2 shows slightly better angular recovery at intermediate radii but similar plateau behavior.

**Interpretation**: Irreducible global dependency is a general PLM phenomenon, but convergence geometry is architecture-dependent.

**Corresponding figure**: Cross-model convergence comparison (Figure 5 in manuscript)

### Robustness Analysis

**Setup**: Validated findings across taxonomic groups, protein families, and window scale variations.

**Finding**: Convergence patterns generalize across diverse protein families (kinases, GPCRs, transcription factors) and taxonomic groups (bacteria, eukaryotes).

**Interpretation**: Irreducible global dependency is a fundamental property of PLM representations, not an artifact of specific protein classes.

**Corresponding figure**: Taxonomy validation results (Figure 6 in manuscript)

---

## Repository Structure

```
P53_Rescue/
├── config.py                    # Centralized configuration (paths, hyperparameters)
├── models.py                    # Shared model classes (LocalSeqEncoder, DeltaDataset)
├── locality_probing/            # Main locality analysis experiments
│   ├── local_encoder/          # Local encoder training scripts
│   ├── p53_analysis/           # P53-specific convergence analysis
│   ├── taxonomy_validation/    # Cross-taxonomy validation
│   ├── structure_analysis/     # Structural metric analysis
│   └── scale_validation/       # Window scale validation
├── cross_model_esm2/            # ESM-2 cross-model validation
│   ├── 02_train_esm2_local_encoder.py    # ESM-2 encoder training
│   ├── 03_analyze_p53_esm2.py             # ESM-2 p53 analysis
│   └── 04_retrain_and_analyze.py         # Retraining experiments
├── representation_analysis/     # Representation-level analyses
│   └── rotation_test/          # Rotation invariance tests
├── data/                        # Processed intermediate data (not committed)
│   ├── embeddings/             # Processed PLM embeddings
│   ├── ppi/                    # Protein-protein interaction data
│   ├── skempi/                 # SKEMPI mutation data
│   ├── p53_mutations/          # P53-specific mutation data
│   ├── esm2/                   # ESM-2 processed data
│   └── pdb/                    # PDB structure files
├── raw_data/                    # Original data sources (not committed)
│   ├── string/                 # STRING database files
│   ├── embeddings/             # Raw PLM embeddings
│   ├── skempi/                 # SKEMPI database
│   └── p53/                    # P53 sequence data
├── results/                     # Generated figures and analysis results
├── legacy/                      # Archived experimental code (not committed)
├── requirements.txt             # Exact package versions for reproducibility
└── .gitignore                   # Git exclusions (data, models, legacy)
```

---

## Reproducibility

### Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install exact package versions
pip install -r requirements.txt
```

### Data Requirements

The repository requires processed mutation datasets with precomputed PLM embeddings. Due to size constraints, these are not included in the repository. To reproduce results:

1. **Download raw data**:
   - STRING PPI: https://string-db.org/cgi/download.pl
   - SKEMPI: https://skempi2.ch.icm.uu.se/
   - ProtT5-XL embeddings: https://github.com/agemagician/ProtTrans
   - ESM-2 embeddings: https://github.com/facebookresearch/esm

2. **Process embeddings** (requires GPU):
   - Extract residue-level embeddings from PLM outputs
   - Generate mutation datasets with local sequence windows
   - Compute global embedding deltas (Δh)

### Training Local Encoders

```bash
# ProtT5-XL encoder
python locality_probing/local_encoder/train_local_encoder.py

# ESM-2 encoder
python cross_model_esm2/02_train_esm2_local_encoder.py
```

Training uses fixed random seeds (`RANDOM_STATE=42` in `config.py`) for reproducibility.

### Reproducing Figures

```bash
# P53 convergence analysis (Figure 2)
python locality_probing/p53_analysis/analyze_convergence.py

# ESM-2 multi-metric analysis (Figure 3-4)
python cross_model_esm2/03_analyze_p53_esm2.py

# Taxonomy validation (Figure 6)
python locality_probing/taxonomy_validation/validate_window_scales.py
```

Figures are saved to the `results/` directory.

### Random Seed Usage

All experiments use fixed random seeds specified in `config.py`:
- `RANDOM_STATE = 42` for train/test splits
- PyTorch random seeds are set before training

### Computational Requirements

- **GPU**: Recommended for PLM embedding extraction and encoder training
- **RAM**: 16GB+ recommended for large embedding datasets
- **Storage**: ~50GB for raw PLM embeddings

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{yang2026locality,
  title={Irreducible Global Dependency in Protein Language Model Representations: Convergence Dynamics and Biological Correlates},
  author={Yang, Jiaxing},
  year={2026},
  journal={bioRxiv},
  doi={10.1101/XXXXXX}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- **ProtT5-XL**: Elnaggar et al., Rost lab (https://github.com/agemagician/ProtTrans)
- **ESM-2**: Meta AI Research (https://github.com/facebookresearch/esm)
- **STRING Database**: STRING Consortium (https://string-db.org)
- **SKEMPI Database**: Jankauskaitė et al. (https://skempi2.ch.icm.uu.se)
- **NMR NOC Data**: Experimental conformational heterogeneity measurements
