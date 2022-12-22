import torch
from Net import Net
from func import download_mnist
import matplotlib.pyplot as plt


# model = torch.load('results/model.pth')
model = Net()
model.load_state_dict(torch.load('results/model.pth'))
model.eval()
data = download_mnist()

with torch.no_grad():
  figure = plt.figure(figsize=(10, 8))
  cols, rows = 5, 5
  for i in range(1, cols * rows + 1):
      sample_idx = torch.randint(len(data[2]), size=(1,)).item()
      img, label = next(iter(data[2]))
      outputs = model(img[0])
      _, predicted = torch.max(outputs.data, 1)
      figure.add_subplot(rows, cols, i)
      plt.title(f'Prediction {predicted.item()}')
      plt.axis("off")
      plt.imshow(img[0].squeeze(), cmap="gray")
plt.show()
plt.savefig("second.jpg")
