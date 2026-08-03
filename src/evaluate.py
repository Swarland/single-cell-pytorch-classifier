import torch
import torch.nn as nn

def evaluate(dataloader, model):

    """ Evaluates a trained neural network. 
   
    Takes a trained neural network model and evaluates it on given dataset
    
    args: 
        dataloader (dataloder cass): PyTorch dataloader class
        model (class): PyTorch nn.model class
    returns:
        accuracy (float): percentage of correct predictions
        average_loss (float): the average Cross Entropy Loss. 
    """

    ## Remember if it was training and restore at end
    was_training = model.training

    ## Set model to evaluate
    model.eval()

    ## Define loss function
    criterion = nn.CrossEntropyLoss()

    ## Initialize variables
    correct_predictions = 0
    total_samples = 0
    total_loss = 0

    ## No longer update gradients
    with torch.no_grad():

        for inputs, targets in dataloader:
            
            ## Get predictions
            outputs = model(inputs)

            ## Calculate Loss
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            ## Get accuracy metrics, F1
            predictions = torch.argmax(outputs, dim = 1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    ## If model was previously training, resore to that state
    if was_training:
        model.train()

    return(accuracy, average_loss)