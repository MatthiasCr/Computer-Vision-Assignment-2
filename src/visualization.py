import matplotlib.pyplot as plt
import numpy as np

def format_positions(positions):
    return ['{0: .3f}'.format(x) for x in positions]


def print_loss(epoch, loss, outputs, target, is_train=True, accuracy=None):
    loss_type = "train loss:" if is_train else "valid loss:"

    if accuracy is not None:
        print("epoch", str(epoch), loss_type, str(loss), "accuracy:", str(accuracy))
    else: 
        print("epoch", str(epoch), loss_type, str(loss))


def plot_loss(epochs, lossesDict):
    plot_x = range(epochs)

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")

    for label, loss in lossesDict.items():
        plt.plot(plot_x, loss,  label = label)    
    
    plt.legend()
    plt.show()


def lidar_to_img(lidar_xyza, figsize=(4, 4)):
    """projects lidar xyza data to 2D image"""

    # move to CPU
    xyza = lidar_xyza.detach().cpu()
    x, y, z, a = xyza

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    a = a.reshape(-1).bool()

    # Color by height
    c = z

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x[a], y[a], z[a], c=c[a], cmap="rainbow", marker="o", s=1,)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-6, 6)
    ax.set_ylim(19, 31)
    ax.set_zlim(-6, 6)
    ax.view_init(elev=0., azim=270)
    ax.set_axis_off()

    # Render to image
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    image = buf[..., :3].copy()

    plt.close(fig)
    return image
