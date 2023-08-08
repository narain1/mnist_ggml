import torch
from torch import nn
import os
import requests
import shutil
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torch.optim import Adam
import struct

mnist_url = 'https://github.com/narain1/dotfiles/releases/download/data/mnist_png.zip'
epochs = 5

def download_file(url):
    resp = requests.get(url, stream=True)
    file_name = url.split('/')[-1]
    os.makedirs('temp', exist_ok=True)
    download_loc = os.path.abspath(os.path.join('temp', file_name))
    with open(download_loc, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk: f.write(chunk)
    return download_loc


def unzip_file(file_location):
    image_root = os.path.join('temp', 'mnist')
    os.makedirs(image_root, exist_ok=True)
    shutil.unpack_archive(file_location, image_root, 'zip')
    return image_root

shutil.rmtree('temp', ignore_errors=True)
p = download_file(mnist_url)
img_loc = unzip_file(p)
train_root = os.path.join(img_loc, 'training')
testing_root = os.path.join(img_loc, 'testing')

# get training image files

def walk_images(path):
    acc = []
    for p, d, fs in os.walk(path):
        for f in fs:
            acc.append(os.path.join(p, f))
    return acc

image_fs = walk_images(train_root)
testing_fs = walk_images(testing_root)

# for p, d, fs in os.walk(train_root):
#     for f in fs:
#         image_fs.append(os.path.join(p, f))

class ImgDataset(Dataset):
    def __init__(self, fs, fold=0, train=True):
        ys = list(map(lambda x: x.split('/')[-2], fs))
        kf = StratifiedKFold(4, random_state=32, shuffle=True)
        idx = list(kf.split(fs, ys))[fold][0 if train else 1]
        self.xs = [fs[i] for i in idx]
        self.ys = [ys[i] for i in idx]

    def __getitem__(self, idx):
        f = self.xs[idx]
        img = np.array(Image.open(f).convert('L')).reshape(-1)
        return torch.from_numpy(img)/255.0, torch.tensor(int(self.ys[idx]))

    def __len__(self):
        return len(self.xs)


class MnistModel(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(inp_dim, 256)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(256, 10)

    def forward(self, xb):
        o = self.l1(xb)
        o = self.act(o)
        return self.l2(o)


train_ds, val_ds = ImgDataset(image_fs), ImgDataset(image_fs, train=False)
train_dl, val_dl = (
    DataLoader(train_ds, shuffle=True, drop_last=True, batch_size=64), 
    DataLoader(val_ds, shuffle=False, batch_size=64)
)
testing_ds = ImgDataset(testing_fs)
testing_dl = DataLoader(testing_ds, shuffle=True, drop_last=True, batch_size=64)

print(f'dataset length train : {len(train_ds)}, val: {len(val_ds)}')
print(f'dataloader length train: {len(train_dl)}, val: {len(val_dl)}')


model = MnistModel(784, 10)
loss_fn = nn.CrossEntropyLoss()
opt = Adam(model.parameters(), lr=1e-4)


for _ in range(epochs):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    for k, (xb, yb) in enumerate(train_dl):
        ys = model(xb)
        acc = (ys.argmax(axis=1) == yb).float().mean()
        train_acc.append(acc.item())
        loss = loss_fn(ys, yb)
        loss.backward()
        opt.step()
        train_loss.append(loss.item())

    with torch.no_grad():
        for k, (xb, yb) in enumerate(val_dl):
            ys = model(xb)
            acc = (ys.argmax(axis=1) == yb).float().mean()
            val_acc.append(acc.item())
            loss = loss_fn(ys, yb)
            val_loss.append(loss.item())

    print(_,
          "{:.4f}".format(np.mean(train_loss)),
          "{:.4f}".format(np.mean(val_loss)),
          "{:.4f}".format(np.mean(train_acc)),
          "{:.4f}".format(np.mean(val_acc))
         )

def validate_model(model):
    with torch.no_grad():
        for k, (xb, yb) in enumerate(testing_dl):
            ys = model(xb)
            acc = (ys.argmax(axis=1) == yb).float().mean()
            val_acc.append(acc.item())
    print("{:.4f}".format(np.mean(val_acc)))


shutil.rmtree('temp')
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/mnist.bin')

state_dict = torch.load('models/mnist.bin', map_location='cpu')
list_vars = state_dict

with open('models/ggml_model.bin', 'wb') as f_out:
    f_out.write(struct.pack('i', 0x67676d6c))

    for name in list_vars.keys():
        data = list_vars[name].squeeze().numpy()
        print("processing variable: " + name + " with shape: ", data.shape)
        n_dims = len(data.shape)
        f_out.write(struct.pack("i", n_dims))

        data = data.astype(np.float32)
        for i in range(n_dims):
            f_out.write(struct.pack("i", data.shape[n_dims - i -1]))

        data.tofile(f_out)

print("Done output file: ", 'models/ggml_model.bin'  )
