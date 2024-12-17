import time

import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)
ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)

batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)
for image_batch, label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break

model = models.MyModel()

acc_train = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_train*100:.3f}%')
acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.3f}%')

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 0.003
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

models.train(model, dataloader_test, loss_fn, optimizer)

acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.3f}%')

n_epochs = 5

for k in range(n_epochs):
    print(f'epoch {k+1}/{n_epochs}', end=': ', flush=True)
    
    time_start = time.time()
    loss_train = models.train(model, dataloader_train, loss_fn, optimizer)
    time_end = time.time()
    print(f'train loss: {loss_train:.3f} ({time_end-time_start}s)')
    print(f'train loss: {loss_train}')
    acc_test = models.test_accuracy(model, dataloader_test)
    print(f'test accuracy: {acc_test*100:.3f}%')
    # 必要なライブラリのインポート
import models

# テストロスを計算
test_loss = models.test(model, dataloader_test, loss_fn)
print(f"Test Loss: {test_loss:.4f}")

import matplotlib.pyplot as plt

# 記録用リスト
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

n_epochs = 5  # エポック数

for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")

    # トレーニング (train loss の計算)
    train_loss = models.train(model, dataloader_train, loss_fn, optimizer)
    train_acc = models.test_accuracy(model, dataloader_train)

    # テスト (test loss の計算)
    test_loss = models.test(model, dataloader_test, loss_fn)
    test_acc = models.test_accuracy(model, dataloader_test)

    # ロスと精度を記録
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # ログを表示
    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")

# エポック数のリスト
epochs = range(1, n_epochs + 1)

# グラフ描画
plt.figure(figsize=(16, 8))#figsizeの変更

# Lossのグラフ
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss", marker='o')
plt.plot(epochs, test_losses, label="Test Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()

# Accuracyのグラフ
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Train Accuracy", marker='o')
plt.plot(epochs, test_accuracies, label="Test Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()

plt.tight_layout()
plt.show()
