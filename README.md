Repository for the source code used to produce the results in the paper **Predicting Treatment Response in Major Depressive Disorder using Structural MRI: A NeuroPharm Study**.

### Summary description of the files

```
├── classification
│   ├── classification.utils.R: Library of useful functions for classification
│   ├── evaluate_generalizability.R: Evaluates generalizability of classifiers (Table S7)
│   ├── nested_cv_metrics_tables.R: Creates Tables 3 & S8
│   ├── nested_cv.R: Perform nested CV of the classifiers for NeuroPharm and EMBARC datasets
│   ├── nested_cv_roc_figures.R: Creates Figures 2 & S1
│   ├── train_final_models.R: Train final models using the whole NeuroPharm dataset
│   ├── vbm_export_classification_data.m: Creates randomized batch of the VBM data
│   ├── vbm_figure.R: Creates Figure S2
│   ├── vbm.R: Trains classifier using VBM data derived from the NeuroPharm dataset
│   └── vbm_table.R: Creates Table S9
├── preprocessing
│   ├── estimate_SPM_ICV.m: Estimation of intracranial volume using SPM/Matlab
│   ├── import_fs_data.py: Export FreeSurfer metrics to CSV files
│   ├── run_recon-all.py: Performs FreeSurfer anatomical reconstruction
│   ├── vbm_processing.m: Performs VBM preprocessing using SPM/Matlab
│   └── vbm_stats.m: Evaluate group difference in VBM metrics
└── stats
    ├── compare_NP1_EMBARC_demographics.R: Compares demographics between NP1 & EMBARC datasets
    ├── demographics_table.R: Creates Tables 1-2 & S1-2
    └── regional_comparison.R: Creates Table S3-6
```
