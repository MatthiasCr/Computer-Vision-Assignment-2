import matplotlib as plt

def format_positions(positions):
    return ['{0: .3f}'.format(x) for x in positions]


def print_loss(epoch, loss, outputs, target, is_train=True, is_debug=False):
    loss_type = "train loss:" if is_train else "valid loss:"
    print("epoch", str(epoch), loss_type, str(loss))
    if is_debug:
        print("example pred:", format_positions(outputs[0].tolist()))
        print("example real:", format_positions(target[0].tolist()))


def plot_loss(epochs, lossesDict):
    plot_x = range(epochs)

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")

    for label, loss in lossesDict.items():
        plt.plot(plot_x, loss,  label = label)    
    
    plt.legend()
    plt.show()