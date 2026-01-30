# Dog vs Cat Image Classifier

I built a convolutional neural network that classifies images as either cats or dogs. After learning about CNNs, I wanted to build something that actually works, so I used transfer learning with MobileNetV2 on 5,000 images.

## The Problem

Distinguish between images of cats and dogs. This is a binary classification problem - each image is either a cat (0) or a dog (1).

## The Data

I used the Bingsu/Cat_and_Dog dataset from Hugging Face. It has 10,000 images total, but I used 5,000 to keep training time reasonable and avoid running out of RAM in Google Colab.

- Training: 4,000 images
- Validation: 1,000 images  
- Test: 1,000 images

All images were resized to 224×224 pixels to match MobileNetV2's expected input size.

## Approach

I used transfer learning instead of training a CNN from scratch. MobileNetV2 was already trained on millions of images, so it already knows how to detect edges, shapes, and patterns. I froze those layers and only trained a new classification layer on top.

I also used data augmentation during training - random rotations, flips, and zooms. This forces the model to learn actual features (like fur patterns and facial structure) instead of memorizing specific orientations.

## Results

The model achieved **98.89% accuracy** on the test set. Out of 992 test images, it only got 11 wrong.

- Cat classification: 488/491 correct (99.4%)
- Dog classification: 493/501 correct (98.4%)

Training took 12 epochs before early stopping kicked in. The model didn't overfit - validation accuracy stayed higher than training accuracy throughout.

## Key Decisions

**Why MobileNetV2?**  
It's lightweight, fast to train, and performs well on image classification. Good balance between accuracy and efficiency.

**Why 5,000 images instead of all 10,000?**  
Google Colab's free tier has limited RAM. 5,000 images is enough to get 98%+ accuracy while staying within memory limits.

**Why these specific augmentations?**  
Cats and dogs can be photographed at different angles, positions, and distances. The augmentations (rotation, flip, shift, zoom) teach the model to recognize animals regardless of how they're positioned in the frame.

## Tools

- Python, TensorFlow/Keras
- Hugging Face datasets library
- MobileNetV2 for transfer learning
- ImageDataGenerator for augmentation
- scikit-learn for train/test split and evaluation metrics


