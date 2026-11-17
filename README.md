# Classifying colon cancer cells in histology images

Convolutional neural networks (CNNs) that classify microscope images of individual colon cells in two ways: whether the cell is cancerous, and what type of cell it is (epithelial, fibroblast, inflammatory or other).

A joint project by Joseph Mchaileh and Robert Harper, completed in 2025 during our Master of Data Science at RMIT University. Built in Python with TensorFlow/Keras and run on Google Colab.

## Data

A version of the CRCHistoPhenotypes dataset of colorectal cancer histology images, from Sirinukunwattana et al. (2016), [*Locality sensitive deep learning for detection and classification of nuclei in routine colon cancer histology images*](https://doi.org/10.1109/TMI.2016.2525803), IEEE Transactions on Medical Imaging.

- 9,896 images of 27×27 pixels from 60 patients, labelled with both cancer status and cell type
- 10,384 further images from 39 patients, labelled with cancer status only

The version used here was prepared for a university course and is not mine to redistribute, so no data is included. The notebook keeps its original outputs.

## Approach

- **Splitting by patient.** Data was split into training (6,778 images), validation (1,738) and test (1,380) sets so that no patient appears in more than one set. A random split would let the model see near-identical cells from the same patient during training and testing, which overstates performance.
- **Class imbalance.** About 42% of cells are cancerous, and cell types are moderately imbalanced. Random oversampling and data augmentation (rotation, flipping) were each tested, alone and combined.
- **Models.** A simple CNN baseline was improved step by step: deeper convolutional blocks, dropout, L2 regularisation, early stopping and learning-rate reduction, with hyperparameters tuned by grid search on the validation set.
- **Metric.** Weighted F1 score, because of the class imbalance. Recall is also reported for the cancer task, where missing a cancerous cell is the costly error.
- **Semi-supervised learning.** The best cell-type model labelled the 10,384 extra images, and a new model was trained on those predictions plus the training set.
- **Held-out test.** The test set was used only once per final model.

## Results

| Task | Validation weighted F1 | Test accuracy | Test weighted F1 |
|---|---|---|---|
| Cancerous or not | 0.908 | 86.4% | 0.865 |
| Cell type | 0.822 | 74.7% | 0.747 |
| Cell type, semi-supervised | 0.814 | 74.4% | 0.743 |

- Data augmentation gave the biggest gain on both tasks. Oversampling helped little and caused earlier overfitting.
- The semi-supervised model did not improve on the final cell-type model.
- Performance on unseen patients was 0.04 to 0.08 below validation, showing a generalisation gap.

## Limitations

- **Cancer status and cell type are entangled.** Every cancerous cell in the data is epithelial, and there are no healthy epithelial cells. The cancer model may partly be learning to recognise epithelial cells rather than cancer.
- **Single cells, no context.** Diagnosis in practice relies on whole-slide images and the surrounding tissue, so this is a long way from a clinical tool.
- **Comparisons with published models are approximate.** The notebook compares results with published models, which score higher, but those studies used different data preparation and splits, and one reports unweighted F1.
- **One hold-out split.** With only 60 labelled patients, results could shift noticeably with a different split of patients.

## Running it

The notebook is set up for Google Colab with a GPU. Save the image data zip to Google Drive, update `zip_path` in the setup cell, then run all cells. To run locally, replace the Drive cell with a local path and install the packages in `requirements.txt`.
