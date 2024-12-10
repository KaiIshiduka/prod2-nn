from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
import torch

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
)

print(len(ds_train))  

image, target = ds_train[0]
print(type(image), target)

plt.imshow(image)
plt.title(target)
plt.show()

image = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image, dtype=torch.float32, scale=True)
print(image.shape, image.dtype)
print(image,min(), image.max())

#適当なコメント