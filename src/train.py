import torch
import torch.nn as nn
from src.evaluate import evaluate

def train(dataloader, model, num_epochs, lr = 0.001, weight_decay = 0, validation_dataloader = None):

    ## Set model to train
    model.train()

    ## Capture loss
    history = {
        "train_loss": [],
        "validation_loss": [],
    }
    ## Define loss function
    criterion = nn.CrossEntropyLoss()

    ## Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    ## Define training loop
    for epoch in range(num_epochs):
        model.train()
        
        total_train_loss = 0

        for inputs, targets in dataloader:

            ## Set gradients to zero
            optimizer.zero_grad()

            ## Run model to get predictions
            outputs = model(inputs)

            ## Compute loss metric 
            loss = criterion(outputs, targets)

            ## Run backward pass to calculate gradients 
            loss.backward()

            ## Update optimizer weights
            optimizer.step()

            ## Record loss
            total_train_loss += loss.item() * inputs.size(0)

        

        average_train_loss = total_train_loss / len(dataloader.dataset)
        history["train_loss"].append(average_train_loss)
        

        

        ## If validation dataloader is inputed function then calls evaluation function
        if validation_dataloader is not None:
            _ , validation_loss = evaluate(validation_dataloader, model)
            history["validation_loss"].append(validation_loss)

        if validation_dataloader is not None:
            print(
                f"Epoch {epoch+1}/{num_epochs} complete.", 
                f"Train loss: {average_train_loss:.4f}",
                f"Validation loss: {validation_loss:.4f}")
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} complete.", 
                f"Train loss: {average_train_loss:.4f}")
    ## Return both training loss and the model
    return model, history