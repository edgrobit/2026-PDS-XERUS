# Projects in Data Science (2026)

Second semester final project of group Xerus.

To optimally run the main, please ensure everything is in correct locations. If run on original data, "masks_without_images.csv" to be left unchanged, else should be omitted or uptaded id's if known.

ProjectInDataScience2026_ExamTemplate/
├── data/
│   ├─ metadata.csv                  #! Clinical metadata
│   ├─ annotations_combined.csv      #! annotations of hair and penmarks 
│   ├─ masks_without_images.csv      #! for any images that have to be omitted. Works only on original data,   │   ...                                 delete otherwise
│   │
│   ├── imgs/                        #! skin images, anything in this folder is git ignored so please upload the images
│   │    ├── img_XX1.png
│   │    ├── img_XX2.png
│   │     ......
│   │    └── img_XXX.png
│   │
│   └── masks/                       #! masks images, anything in this folder is git ignored so please upload the images
│        ├── mask_XX1.png
│        ├── mask_XX2.png
│         ......
│        └── mask_XXX.png
│
├── src/
│   ├── __init__.py
│   ├── feature_A.py                     
│   ├── feature_B.py                    
│   ......
│   └── feature_X.py                    
│ 
├── result/
│   ├── figures/                        # Figures, product of model output
│   ├── models/                         # Trained models
│   ├── predictions/                    # Probabilities outputed by the models
│   └── reports                         # Files related to the Mandatory assignment
│        ├── report_GROUPEID.pdf
│        └── features_GROUPEID.csv
│ 
├── main.py                             # script to train or evaluate models
└── README.md
```
