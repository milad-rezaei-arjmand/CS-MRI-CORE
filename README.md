# CS-MRI-CORE

## Parameter-Efficient Cross-Slice Adaptation of an MRI Foundation Model for Extreme Few-Shot Meningioma Segmentation

CS-MRI-CORE is a research project focused on **MRI foundation models**, **extreme few-shot medical image segmentation**, and **parameter-efficient adaptation** for meningioma segmentation.

> **Manuscript status:** Submitted for peer review.

The current public repository is intended to document the project and its research scope. The full experimental implementation, medical imaging data, and trained checkpoints are not included in the present public release.

---

## Research Focus

The project investigates how a pretrained MRI foundation model can be adapted to a target segmentation task when only a very limited amount of labeled data is available.

The main research themes are:

- MRI foundation models
- Few-shot medical image segmentation
- Parameter-efficient adaptation
- Cross-slice contextual information
- Meningioma segmentation
- Data-efficient deep learning

---

## Motivation

Medical image segmentation models often depend on substantial amounts of expert-annotated data. In clinical imaging, obtaining high-quality segmentation labels can be expensive and time-consuming.

CS-MRI-CORE studies a more data-efficient setting by combining a pretrained MRI representation with a lightweight adaptation strategy and contextual information from neighboring slices.

The objective is to investigate whether useful task adaptation can be achieved without full model fine-tuning under extreme few-shot supervision.

---

## Conceptual Pipeline

```text
MRI Input
   |
   v
Target Slice + Cross-Slice Context
   |
   v
Pretrained MRI Foundation Model
   |
   v
Parameter-Efficient Adaptation
   |
   v
Few-Shot Segmentation
   |
   v
Meningioma Mask Prediction
   |
   v
Evaluation
```

---

## Repository Status

This repository currently contains the public-facing structure and documentation for the project.

### Publicly available

- Project description
- Dependency list
- Dataset placeholder/documentation
- Checkpoint placeholder/documentation
- License

### Not included in the current public release

- Original medical imaging data
- Trained model checkpoints
- Full training implementation
- Full evaluation implementation
- Manuscript source files

The repository may be expanded after the peer-review process, subject to publication and data-usage requirements.

---

## Repository Structure

```text
CS-MRI-CORE/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   └── README.md
└── checkpoints/
    └── README.md
```

---

## Dependencies

The current environment specification includes:

```text
torch
torchvision
numpy
pandas
opencv-python
scikit-learn
matplotlib
tqdm
nibabel
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Data Availability

Medical imaging data are not distributed through this repository.

Dataset access, preparation, and use must follow the terms and policies of the corresponding data provider.

---

## Model Checkpoints

Trained checkpoints are not included in the current public release.

If checkpoints are released later, download instructions and model-loading details should be documented in this repository.

---

## Publication

**CS-MRI-CORE: Parameter-Efficient Cross-Slice Adaptation of an MRI Foundation Model for Extreme Few-Shot Meningioma Segmentation**

**Status:** Submitted manuscript / under peer review

Citation information will be added if and when the manuscript is published.

---

## Author

**Milad Rezaei Arjmand**

M.Sc. Student in Biomedical Engineering (Bioelectric)  
Research interests: Medical Artificial Intelligence, Medical Imaging, Foundation Models, Medical Image Segmentation, and Data-Efficient Deep Learning.

---

## License

See the `LICENSE` file included in this repository.

The license applies to repository content only. Any external datasets remain subject to their original licenses and terms of use.
