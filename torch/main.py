import torch
import torchvision
import matplotlib.pyplot as plt
from func import download_mnist


batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)


def show(train_data):

    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = next(iter(train_data))
        figure.add_subplot(rows, cols, i)
        plt.title(label[0])
        plt.axis("off")
        plt.imshow(img[0].squeeze(), cmap="gray")
    plt.show()
    plt.savefig("first.jpg")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
data = download_mnist()
show(data[0])
