## Horse vs Human (Binary Recognition)

**Source:**
https://www.kaggle.com/rishitdagli/horses-vs-humans


**Info:**
Binary Classification,
- Training: 500 Horses, 527 Human, 
- Testing: 128 Horses, 128 Human.

**Version and performance (Test):**
- **CNN** max(**50 epochs** vs Callbacks 0.98) 0.8867
- **CNN + Data Augmentation** (no good for this dataset) no stable
- **CNN + Data Augmentation + Transfer Learning** 0.9844

