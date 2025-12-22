from visualization import print_loss
import wandb


def initWandbRun(fusion_type, epochs, batch_size, parameters):
    return wandb.init(
    entity="matthiascr-hpi-team",
    project="cilp-extended-assessment",
    config={
        "fusion type": fusion_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "parameters": parameters,
    },
)


def train_model(model, optimizer, scheduler, loss_func, epochs, train_dataloader, valid_dataloader, device, wandbRun):
    train_losses = []
    valid_losses = []
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
            scheduler.step()
            train_loss += loss.item()

        train_loss = train_loss / (step + 1)
        train_losses.append(train_loss)
        print_loss(epoch, train_loss, outputs, target, is_train=True)
        
        model.eval()
        valid_loss = 0
        for step, batch in enumerate(valid_dataloader):
            target = batch[2].to(device)
            inputs_rgb = batch[0].to(device)
            inputs_xyz = batch[1].to(device)
            outputs = model(inputs_rgb, inputs_xyz)
            valid_loss += loss_func(outputs, target).item()
        valid_loss = valid_loss / (step + 1)
        valid_losses.append(valid_loss)
        print_loss(epoch, valid_loss, outputs, target, is_train=False)

        wandbRun.log({'train_loss': train_loss, 'valid_loss': valid_loss})
    return train_losses, valid_losses
