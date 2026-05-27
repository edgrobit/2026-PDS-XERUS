# Projects in Data Science (2026)

Second semester final project of group Xerus.

To optimally run the main, please ensure everything is in correct locations. If run on original data, "masks_without_images.csv" to be left unchanged, else should be omitted or uptaded id's if known.

```text
├── data/
│   ├── metadata.csv                  # Clinical metadata
│   ├── annotations_combined.csv      # Annotations of hair and penmarks
│   ├── masks_without_images.csv      # Images that should be omitted
│   │
│   ├── imgs/                         # Skin lesion images (git ignored)
│   │   ├── img_XX1.png
│   │   ├── img_XX2.png
│   │   ├── ...
│   │   └── img_XXX.png
│   │
│   └── masks/                        # Mask images (git ignored)
│       ├── mask_XX1.png
│       ├── mask_XX2.png
│       ├── ...
│       └── mask_XXX.png
│
├── src/
│   ├── __init__.py
│   ├── feature_A.py
│   ├── feature_B.py
│   ├── ...
│   └── feature_X.py
│
├── result/
│   ├── figures/                      # Model output figures
│   ├── models/                       # Trained models
│   ├── predictions/                  # Model prediction probabilities
│   └── reports/
│       ├── report_GROUPID.pdf
│       └── features_GROUPID.csv
│
├── main.py                           # Train/evaluate models
└── README.md
```