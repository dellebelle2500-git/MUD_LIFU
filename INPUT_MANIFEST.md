# Input manifest

The figure scripts expect the following repository-relative inputs.

## `data/`

- `normal_subject_feature_matrix_clean.csv`
- `baseline_subject_feature_matrix_new.csv`
- `post_7d_subject_feature_matrix_new.csv`
- `post_3M_subject_feature_matrix_new.csv`
- `subject_cluster_new.csv`
- `FDCR_Final_45ROIs_Mean_Features.csv`
- `FDCR_main_effect_activation.xlsx`
- `seedtovoxel_ANOVA_main_effect_longitudinal_final.xlsx`
- `subject_VAS_final.csv`
- `clinical_scales_deident.csv`

## `rasters/`

- `Fig1a.png`
- `Fig2f_AG_Rt.png`
- `Fig2f_PCC.png`
- `Fig2f_PHC_Lt.png`
- `Fig2f_Precuneus.png`
- `Fig3b_dlPFC.png`
- `Fig3b_IFC.png`
- `Fig3b_antPFC.png`
- `Fig3b_OFC.png`
- `Fig3b_PPC.png`
- `Fig3b_MTG.png`
- `circles.json`

The MRI/statistical-map raster files are the no-circle inputs. ROI rings are reconstructed by `helpers/fig_rasters.py` from `circles.json`.
