import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from Net import Net
from func import download_mnist
import matplotlib.pyplot as plt


model = Net()
data = download_mnist()
with torch.no_grad():
  output = model(data[1])


figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(data[1]), size=(1,)).item()
    img, label = next(iter(data[0]))
    figure.add_subplot(rows, cols, i)
    plt.title("Prediction: {}".format(
      output.data.max(1, keepdim=True)[1][i].item()))
    plt.axis("off")
    plt.imshow(img[0].squeeze(), cmap="gray")
plt.show()
plt.savefig("second.jpg")

# model = Net()
# main = model.load_state_dict(torch.load('./results/model.pth'))
# model.eval()
# print(model)