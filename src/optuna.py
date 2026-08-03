import optuna 

from src.train import train
from src.evaluate import evaluate
from src.model import SingleCellModel
from src.utils import set_seed

def tune(X_tensor, class_names, train_dataloader, validate_dataloader, num_epochs, num_trials):

    """ Tunes a neural network model using the optuna package. 
   
    Takes a neural network model and finds best hyperparameters for specific model used. 
    
    args:
        X_tensor (tensor): original tensor used for data creation. Number of features in data derived from it.
        class_names (list): list of class names in dataset. Number of classes derived from it.
        train_dataloader (dataloder cass): PyTorch dataloader class containing training data
        validate_dataloader (dataloder cass): PyTorch dataloader class containing  validation data
        num_epochs (integer): number of epoch to train
        num_trials (integer): number of trials to test hyperparameters in

    returns:
        optuna study
    """

    def objective(trial):
        set_seed(42)
        ## Define hyperparameter space
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log = True)
        weight_decay =  trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        hidden_size1 = trial.suggest_categorical("hidden_size1", [64, 128, 256])
        hidden_size2 = trial.suggest_categorical("hidden_size2", [32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)
        

        ## Call model 
        model = SingleCellModel(
        num_features=X_tensor.shape[1],
        num_classes=len(class_names),
        hidden_size1 = hidden_size1, 
        hidden_size2 = hidden_size2,
        dropout_rate = dropout_rate)

        ## Train model
        train(train_dataloader, model, num_epochs = num_epochs, lr = lr, weight_decay = weight_decay)

        _ , average_loss = evaluate(validate_dataloader, model)

        return average_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)

    return study