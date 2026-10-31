# ECE1512 Project A: Multiple Instance Learning for Whole-Slide Histopathology

This project implements and evaluates **Attention-Based Multiple Instance Learning (ABMIL)** for whole-slide histopathology classification using PyTorch.

Starting from the instructor-provided ABMIL framework, we investigate an extension based on **Attention-Challenging Multiple Instance Learning (ACMIL)**, focusing on multi-branch attention aggregation, attention masking, and class-imbalance-aware classification.

Experiments are conducted on three datasets: CAMELYON16 (C16), CAMELYON17 (C17), and BRACS.

## 1. Dataset Setup

The pre-extracted histopathology features are available on Zenodo:

https://zenodo.org/records/23130543

Download the HDF5 feature files and place them in the following directories:

```text
feature/
  C16/patch_feats_pretrain_medical_ssl.h5
  C17/patch_feats_pretrain_medical_ssl.h5
  BRACS/patch_feats_pretrain_medical_ssl.h5
```

If the feature files are stored elsewhere, update `data_dir` in the corresponding YAML configuration.

Each HDF5 file contains slide-level entries with patch features, optional spatial coordinates, and integer class labels.

## 2. Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The notebook can be executed locally, on a GPU server, or in Google Colab.

Set `PROJECT_DIR` in `ECE1512_ProjectA_Student.ipynb` to the project directory before running the notebook.

## 3. Implementation

The completed implementation is contained in:

```text
ECE1512_ProjectA_Student.ipynb
```

The project retains the instructor-provided data loading and evaluation infrastructure:

- `dataloader.py`: HDF5 loading, dataset splitting, and MIL batching.
- `utils.py`: Configuration loading, random seeding, and evaluation metrics.
- `config/`: Dataset-specific configurations.
- `dataset_csv/`: Metadata used for dataset splitting.

The notebook contains the ABMIL baseline, the proposed multi-branch extension, training and validation procedures, experimental comparisons, and final evaluation.

### ABMIL Baseline

The baseline uses gated attention to aggregate pretrained patch features into a slide-level representation, followed by a classification layer.

### Multi-Branch Attention Extension

The proposed model extends ABMIL with multiple attention branches, following ACMIL (Zhang et al., ECCV 2024).

Each attention branch produces a separate slide representation and receives auxiliary classification supervision. Pairwise cosine similarity is used to encourage attention diversity.

We investigate three aspects of the model:

1. **Aggregation and masking:** Mean versus concatenation aggregation, with and without Stochastic Top-K Instance Masking (STKIM).
2. **Number of attention heads:** Comparison of 4, 5, and 6 attention heads.
3. **Classification objectives:** Standard cross-entropy, Balanced Softmax, and Class-Balanced Cross-Entropy, evaluated on C17 and BRACS.

The concatenation-based aggregation and classification-loss comparisons are experimental modifications to the ACMIL framework.

## 4. Running the Experiments

Open `ECE1512_ProjectA_Student.ipynb`.

**ABMIL baseline:** Select the dataset using `DATASET` (`C16`, `C17`, or `BRACS`) and execute the baseline training and evaluation cells. Repeat for all three datasets, including the Additional Analysis saving cell, to retain their results.

**Research extension:** Execute the extension cells in notebook order. The experimental loops automatically iterate over the required datasets and configurations, without manually switching `DATASET`.

Each experimental stage selects its preferred configuration using validation AUROC and macro-F1. The selected configurations are subsequently evaluated on the test partitions.

The notebook includes the corresponding results tables and visualizations.

## 5. Saved Results

Results are organized as follows:

```text
results/
  baseline/
    C16/
    C17/
    BRACS/
  extension/
    C16/
    C17/
    BRACS/
    validation_summary.csv
    final_choices.csv
    test_summary.csv
```

The dataset-specific directories contain saved model checkpoints, training histories, and predictions.

The summary files contain the validation experiment results, selected model configurations, and final test performance.

The notebook also generates plots for validation comparisons, training histories, and confusion matrices.

## 6. Report and References

The accompanying **Project A report** provides the research motivation, proposed ideas, methodology, experimental results, comparisons with ABMIL, and discussion of limitations.

For quantitative results, interpretation of the experiments, and conclusions, refer to the report.

The primary methodological references are:

1. Ilse et al., *Attention-based Deep Multiple Instance Learning*, ICML 2018.
2. Zhang et al., *Attention-Challenging Multiple Instance Learning*, ECCV 2024.
3. Ren et al., *Balanced Meta-Softmax for Long-Tailed Visual Recognition*, NeurIPS 2020.
4. Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019.
