import torch
from Net import Net
from PIL import Image
from torchvision import transforms


model = Net()
model.load_state_dict(torch.load('results/model.pth'))
model.eval()
# image = img.imread('1.png')

image = Image.open('1.png')
image = image.convert('1')

transform = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

img_tensor = transform(image)
print(img_tensor)
outputs = model(img_tensor)

# with torch.no_grad():
#   figure = plt.figure(figsize=(10, 8))
#   cols, rows = 5, 5
#   for i in range(1, cols * rows + 1):
#       sample_idx = torch.randint(len(data[2]), size=(1,)).item()
#       img, label = next(iter(data[2]))
#       outputs = model(img[0])
#       _, predicted = torch.max(outputs.data, 1)
#       figure.add_subplot(rows, cols, i)
#       plt.title(f'Prediction {predicted.item()}')
#       plt.axis("off")
#       plt.imshow(img[0].squeeze(), cmap="gray")
# plt.show()
# plt.savefig("second.jpg")