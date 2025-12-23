from visualization import print_loss, lidar_to_img
from utils import set_seeds
import torch
import wandb


def initWandbRun(fusion_type, embedding_size, epochs, batch_size, parameters, optimizer, lr_scheduler, lr_start, lr_end):
    return wandb.init(
    entity="matthiascr-hpi-team",
    project="cilp-extended-assessment",
    config={
        "fusion type": fusion_type,
        "embedding_size": embedding_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "parameters": parameters,
        "optimizer": optimizer,
        "lr scheduler": lr_scheduler,
        "lr_start": lr_start,
        "lr_end": lr_end
    },
)

def get_correct(output, y, device):
    zero_tensor = torch.tensor([0]).to(device)
    pred = torch.gt(output, zero_tensor)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct


def train_model(model, optimizer, loss_func, epochs, train_dataloader, valid_dataloader, device, wandbRun, scheduler=None, output_name="best_model"):
    set_seeds(51)
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
            target = batch[2].to(device)
            inputs_rgb = batch[0].to(device)
            inputs_xyz = batch[1].to(device)
            outputs = model(inputs_rgb, inputs_xyz)
            
            loss = loss_func(outputs, target)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            train_loss += loss.item()

        train_loss = train_loss / (step + 1)
        train_losses.append(train_loss)
        print_loss(epoch, train_loss, outputs, target, is_train=True)
        
        model.eval()
        valid_loss = 0
        correct = 0
        for step, batch in enumerate(valid_dataloader):
            target = batch[2].to(device)
            inputs_rgb = batch[0].to(device)
            inputs_xyz = batch[1].to(device)
            outputs = model(inputs_rgb, inputs_xyz)
            valid_loss += loss_func(outputs, target).item()
            correct += get_correct(outputs, target, device)
        valid_loss = valid_loss / (step + 1)
        valid_losses.append(valid_loss)
        accuracy = correct/valid_N
        print_loss(epoch, valid_loss, outputs, target, is_train=False, accuracy=accuracy)

        wandbRun.log(
            {
                'train_loss': train_loss, 
                'valid_loss': valid_loss, 
                'valid_accuray': accuracy,
                'learning_rate': scheduler.get_last_lr()[0]
            }
        )

        # checkpointing
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model = model
            # Save the best model
            torch.save(best_model.state_dict(), model_save_path)
            print('Found and saved better weights for the model')

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
