# Train.py

"""

Entrenar desde 0.
Feature extraction
Fine-tuning

Steps:
1. Load data
2. Define the model
3. Define the loss function
4. Define the optimizer
5. Train the model
6. Evaluate the model


1. Load data - Transforms & Dataloader
2. Load resnet18(pretrained=True) - Train head 
3. Unfreeze layers and continue training
4. Compare losses and accuracy - Confusion matrix
5. Save best model

"""