import torch
import torch.nn as nn

def train(dataloader, model, num_epochs, lr = 0.001, weight_decay = 0):

    ## Set model to train
    model.train()

    ## Define loss function
    criterion = nn.CrossEntropyLoss()

    ## Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    ## Define training loop
    for epoch in range(num_epochs):

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
        print(
            f"Epoch {epoch+1}/{num_epochs} complete.", 
            f"Loss: {loss.item():.4f}")