import torch
import wandb
import time
import random
from .visualization import print_loss, lidar_to_img
from .utils import set_seeds

WANDB_TEAM_NAME = "matthiascr-hpi-team"
WANDB_PROJECT_NAME = "cilp-extended-assessment"

def initWandbRun(fusion_type, epochs, batch_size, parameters, optimizer, lr_scheduler, lr_start, lr_end):
    run = wandb.init(
    entity = WANDB_TEAM_NAME,
    project = WANDB_PROJECT_NAME,
    config = {
        "fusion type": fusion_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "parameters": parameters,
        "optimizer": optimizer,
        "lr scheduler": lr_scheduler,
        "lr_start": lr_start,
        "lr_end": lr_end
    })

    # for valid loss and accuracy log the best values as summary
    run.define_metric("valid_loss", summary="min")
    run.define_metric("valid_accuracy", summary="max")
    return run


def get_correct(output, y, device):
    zero_tensor = torch.tensor([0]).to(device)
    pred = torch.gt(output, zero_tensor)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct


def train_model(
        model, 
        optimizer, 
        apply_model, 
        loss_func, 
        epochs, 
        train_dataloader, 
        valid_dataloader, 
        device, 
        wandbRun, 
        scheduler=None, 
        output_name="best_model",
        calc_accuracy=True
    ):
    """
    args:
        apply_model: function that that applies the model on a batch and returns outputs and target
        loss_func: function that calculates loss given model outputs and target.
        wandbRun: wandb run to log metrics to
        scheduler: learning rate scheduler. Can be None then constant learning rate defined in the optimizer is used
        output_name: name under which the best models parameters are stored
        calc_accuracy: wether accuracy should be calculated during validation. 
            Set this to False when its not a classification task.
    """
    saved_random_state = set_seeds(51)

    # start timer
    train_start_time = time.perf_counter()
    
    train_losses = []
    valid_losses = []
    valid_N = len(valid_dataloader.dataset)

    best_val_loss = float('inf')
    best_model = None
    model_save_path = f"../checkpoints/{output_name}.pt"

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs, target = apply_model(model, batch)
            loss = loss_func(outputs, target)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            train_loss += loss.item()

        train_loss = train_loss / (step + 1)
        train_losses.append(train_loss)
        print_loss(epoch, train_loss, is_train=True)
        
        model.eval()
        valid_loss = 0
        correct = 0
        for step, batch in enumerate(valid_dataloader):
            outputs, target = apply_model(model, batch)
            loss = loss_func(outputs, target)
            valid_loss += loss.item()
            if calc_accuracy:
                correct += get_correct(outputs, target, device)
        valid_loss = valid_loss / (step + 1)
        valid_losses.append(valid_loss)
        
        # logging
        log_dict = {}

        if calc_accuracy:
            # since the dataloader have drop_last = True we only predict on 384 samples out of the 400
            # For now just hardcoded
            accuracy = correct / 384 # instead of valid_N
            print_loss(epoch, valid_loss, is_train=False, accuracy=accuracy)
            log_dict['valid_accuracy'] = accuracy
        else:
            print_loss(epoch, valid_loss, is_train=False)

        if scheduler is not None:
            log_dict['learning_rate'] = scheduler.get_last_lr()[0]

        log_dict['train_loss'] = train_loss
        log_dict['valid_loss'] = valid_loss
        wandbRun.log(log_dict)

        # checkpointing
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model = model
            # Save the best model
            torch.save(best_model.state_dict(), model_save_path)
            print('Found and saved better weights for the model')

    # end timer and log total train time
    total_train_time_sec = time.perf_counter() - train_start_time
    wandbRun.summary["total_train_time_sec"] = total_train_time_sec
    
    random.setstate(saved_random_state)
    random.seed(None)

    return train_losses, valid_losses


def log_predictions(model, dataloader, device, run, num_batches):
    """
    Conducts inference for num_batches of dataloader. 
    Logs images of rgb and lidar together with the label, prediction, and prediction probability to the given wandb run
    """
    model.eval()
    table = wandb.Table(
        columns=["rgb", "lidar", "label", "prediction", "probability"]
    )

    with torch.no_grad():
        for i, (rgb, lidar_xyza, label) in enumerate(dataloader):
            if i >= num_batches:
                break
    
            rgb = rgb.to(device)
            lidar_xyza = lidar_xyza.to(device)
            logits = model(rgb, lidar_xyza)
            
            # apply sigmoid to get probability then use threshold 0.5 for class decision
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs >= 0.5).long()

            for b in range(rgb.shape[0]):
                rgb_img = (rgb[b].permute(1, 2, 0).cpu().numpy())
                
                # Produde 2D image of lidar xyza data
                lidar_img = lidar_to_img(lidar_xyza[b])

                table.add_data(
                    wandb.Image(rgb_img, caption="RGB"),
                    wandb.Image(lidar_img, caption="LiDAR"),
                    label[b].item(),
                    preds[b].item(),
                    probs[b].item(),
                )
    run.log({"predictions": table})
