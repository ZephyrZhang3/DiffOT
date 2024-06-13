import gc

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .fid_score import calculate_frechet_distance

from .distributions import DatasetSampler, LoaderSampler
from .inception import InceptionV3


def get_random_colored_images(images, seed=0x000000):
    np.random.seed(seed)

    images = 0.5 * (images + 1)
    size = images.shape[0]
    colored_images = []
    hues = 360 * np.random.rand(size)

    for V, H in zip(images, hues):
        V_min = 0

        a = (V - V_min) * (H % 60) / 60
        V_inc = a
        V_dec = V - a

        colored_image = torch.zeros((3, V.shape[1], V.shape[2]))
        H_i = round(H / 60) % 6

        if H_i == 0:
            colored_image[0] = V
            colored_image[1] = V_inc
            colored_image[2] = V_min
        elif H_i == 1:
            colored_image[0] = V_dec
            colored_image[1] = V
            colored_image[2] = V_min
        elif H_i == 2:
            colored_image[0] = V_min
            colored_image[1] = V
            colored_image[2] = V_inc
        elif H_i == 3:
            colored_image[0] = V_min
            colored_image[1] = V_dec
            colored_image[2] = V
        elif H_i == 4:
            colored_image[0] = V_inc
            colored_image[1] = V_min
            colored_image[2] = V
        elif H_i == 5:
            colored_image[0] = V
            colored_image[1] = V_min
            colored_image[2] = V_dec

        colored_images.append(colored_image)

    colored_images = torch.stack(colored_images, dim=0)
    colored_images = 2 * colored_images - 1

    return colored_images


def h5py_to_dataset(path, img_size=64):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = (
            2
            * (torch.tensor(np.array(data), dtype=torch.float32) / 255.0).permute(
                0, 3, 1, 2
            )
            - 1
        )
        dataset = F.interpolate(dataset, img_size, mode="bilinear")

    return TensorDataset(dataset, torch.zeros(len(dataset)))


class ZeroImageDataset(Dataset):
    def __init__(self, n_channels, h, w, n_samples, transform=None):
        self.n_channels = n_channels
        self.h = h
        self.w = w
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.ones(self.n_channels, self.h, self.w), torch.zeros(
            self.n_channels, self.h, self.w
        )


def load_dataset(
    name,
    path,
    img_size=64,
    batch_size=64,
    shuffle=True,
    device="cuda",
    return_dataset=False,
    num_workers=0,
):
    if name in ["CelebA_low", "CelebA_high"]:
        res = name.split("_")[1]

        if res == "high":
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.CenterCrop(140),
                    torchvision.transforms.Resize((64, 64)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: 2 * x - 1),
                ]
            )
        elif res == "low":
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.CenterCrop(140),
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.Resize((64, 64)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: 2 * x - 1),
                ]
            )

        dataset = ImageFolder(path, transform=transform)

        train_ratio = 0.45
        test_ratio = 0.1

        train_size = int(len(dataset) * train_ratio)
        test_size = int(len(dataset) * test_ratio)
        idx = np.arange(len(dataset))

        np.random.seed(0x000000)
        np.random.shuffle(idx)

        if res == "low":
            train_idx = idx[:train_size]
        elif res == "high":
            train_idx = idx[train_size:-test_size]
        test_idx = idx[-test_size:]

        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)

    elif name.startswith("MNIST"):
        # In case of using certain classe from the MNIST dataset you need to specify them by writing in the next format "MNIST_{digit}_{digit}_..._{digit}"
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: 2 * x - 1),
            ]
        )

        dataset_name = name.split("_")[0]
        is_colored = dataset_name[-7:] == "colored"

        classes = [int(number) for number in name.split("_")[1:]]
        if not classes:
            classes = [i for i in range(10)]

        train_set = datasets.MNIST(path, train=True, transform=transform, download=True)
        test_set = datasets.MNIST(path, train=False, transform=transform, download=True)

        train_test = []

        for dataset in [train_set, test_set]:
            data = []
            labels = []
            for k in range(len(classes)):
                data.append(
                    torch.stack(
                        [
                            dataset[i][0]
                            for i in range(len(dataset.targets))
                            if dataset.targets[i] == classes[k]
                        ],
                        dim=0,
                    )
                )
                labels += [k] * data[-1].shape[0]
            data = torch.cat(data, dim=0)
            data = data.reshape(-1, 1, 32, 32)
            labels = torch.tensor(labels)

            if is_colored:
                data = get_random_colored_images(data)

            train_test.append(TensorDataset(data, labels))

        train_set, test_set = train_test
    else:
        raise Exception("Unknown dataset")

    if return_dataset:
        return train_set, test_set

    train_sampler = LoaderSampler(
        DataLoader(
            train_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
        ),
        device,
    )
    test_sampler = LoaderSampler(
        DataLoader(
            test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
        ),
        device,
    )
    return train_sampler, test_sampler


def ewma(x, span=200):
    return pd.DataFrame({"x": x}).ewm(span=span).mean().values[:, 0]


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)


def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
    elif classname.find("BatchNorm") != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())

def get_loader_stats(XY_sampler, size, batch_size=8, inception=False, use_Y=False, resize_to_64=False, verbose=False):
    if inception:
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).cuda()
        freeze(model)
    else:
        X, Y = XY_sampler.sample(1)
        X, Y = X.flatten(0, 1), Y.flatten(0, 1)
        if use_Y:
            dims = np.prod(Y.shape)
        else:
            dims = np.prod(X.shape)

    pred_arr = np.empty((size, dims))
   
    with torch.no_grad():
        for i in tqdm(range(0, size, batch_size)) if verbose else range(0, size, batch_size):
            start, end = i, min(i + batch_size, size)

            X, Y = XY_sampler.sample(end-start)
            X, Y = X.flatten(0, 1), Y.flatten(0, 1)
            if use_Y:
                batch = ((Y + 1) / 2).type(torch.FloatTensor).cuda()
            else:
                batch = ((X + 1) / 2).type(torch.FloatTensor).cuda()

            assert batch.shape[1] in [1, 3]
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            if inception:
                pred = model(batch)[0]
            else:
                pred = batch
            
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma


def get_Z_pushed_loader_stats(
    T,
    loader,
    ZC=1,
    Z_STD=0.1,
    batch_size=8,
    n_epochs=1,
    verbose=False,
    device="cuda",
    use_downloaded_weights=False,
):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(
        device
    )
    freeze(model)
    freeze(T)

    size = len(loader.dataset)
    pred_arr = []

    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in (
                enumerate(loader) if not verbose else tqdm(enumerate(loader))
            ):
                Z = torch.randn(len(X), ZC, 1, 1) * Z_STD
                XZ = (X, Z)
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    batch = (
                        T(
                            XZ[0][start:end].type(torch.FloatTensor).to(device),
                            XZ[1][start:end].type(torch.FloatTensor).to(device),
                        )
                        .add(1)
                        .mul(0.5)
                    )
                    pred_arr.append(
                        model(batch)[0].cpu().data.numpy().reshape(end - start, -1)
                    )

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect()
    torch.cuda.empty_cache()
    return mu, sigma


def get_enot_sde_pushed_loader_stats(
    T,
    loader,
    batch_size=8,
    n_epochs=1,
    verbose=False,
    device="cuda",
    use_downloaded_weights=False,
):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(
        device
    )
    freeze(model)

    size = len(loader.dataset)
    pred_arr = []

    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in (
                enumerate(loader) if not verbose else tqdm(enumerate(loader))
            ):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    batch = (
                        T(X[start:end].type(torch.FloatTensor).to(device))[0]
                        .add(1)
                        .mul(0.5)
                    )

                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    pred_arr.append(
                        model(batch)[0].cpu().data.numpy().reshape(end - start, -1)
                    )

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect()
    torch.cuda.empty_cache()
    return mu, sigma


def get_sde_pushed_loader_stats(
    T,
    loader,
    batch_size=8,
    n_epochs=1,
    verbose=False,
    device="cuda",
    use_downloaded_weights=False,
):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(
        device
    )
    freeze(model)
    freeze(T)

    size = len(loader.dataset)
    pred_arr = []

    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in (
                enumerate(loader) if not verbose else tqdm(enumerate(loader))
            ):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    batch = (
                        T(X[start:end].type(torch.FloatTensor).to(device))[0][:, -1]
                        .add(1)
                        .mul(0.5)
                    )

                    assert batch.shape[1] in [1, 3]
                    if batch.shape[1] == 1:
                        batch = batch.repeat(1, 3, 1, 1)
                    pred_arr.append(
                        model(batch)[0].cpu().data.numpy().reshape(end - start, -1)
                    )

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect()
    torch.cuda.empty_cache()
    return mu, sigma


def linked_mapping(SDEs, x, return_trajectory=False):
    for sde in SDEs:
        freeze(sde)

    tr = []
    y = x
    for sde in SDEs:
        trajectory, times, shifts = sde(y)
        y = trajectory[:, -1]
        if return_trajectory:
            tr.append(y.clone().detach())
    if return_trajectory:
        return tr
    
    return y

def get_linked_sdes_pushed_loader_stats(
        SDEs, 
        XY_sampler, 
        size, 
        batch_size=8, 
        inception=False, 
        resize_to_64=False, 
        verbose=False,
        device="cuda",
        use_downloaded_weights=False,
):
    XY_sampler.get_labels=False
    if inception:
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(
            device
        )
        freeze(model)
    else:
        X, _ = XY_sampler.sample(1)

        dims = np.prod(X.shape)

    pred_arr = np.empty((size, dims))
   
    with torch.no_grad():
        for i in tqdm(range(0, size, batch_size)) if verbose else range(0, size, batch_size):
            start, end = i, min(i + batch_size, size)

            X, _ = XY_sampler.sample(end-start)
            X = X.flatten(0, 1)
            T_X = linked_mapping(SDEs, X)

            batch = ((T_X + 1) / 2).type(torch.FloatTensor).cuda()
                
            assert batch.shape[1] in [1, 3]
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            if inception:
                pred = model(batch)[0]
            else:
                pred = batch
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

def test_accuracy(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            x, y = data
            outputs = model(x.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y.cuda()).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network:', accuracy)
    return accuracy

def compute_transport_accuracy(SDEs, XY_test_sampler, batch_size, classifier):
    transport_results = []
    real_labels = []
    flat_transport_results =[]
    flat_real_labels = []

    XY_test_sampler.get_labels=True
    for i in range(0, len(XY_test_sampler), batch_size):
        X, _, labels = XY_test_sampler.sample(batch_size)
        X = X.flatten(0, 1)
        real_labels.append(labels)
        
        with torch.no_grad():
            T_X = linked_mapping(SDEs, X.cuda())
            transport_results.append(T_X)

    for sublist, ys in zip(transport_results, real_labels):
        for item, y in zip(sublist, ys):
            flat_transport_results.append(item.data.cpu().numpy())
            flat_real_labels.append(y)

    flat_transport_results = torch.Tensor(flat_transport_results)
    flat_real_labels = torch.LongTensor(flat_real_labels)
    transport_dataset = torch.utils.data.TensorDataset(flat_transport_results, flat_real_labels)
    transport_loader = torch.utils.data.DataLoader(transport_dataset, batch_size=100, shuffle=False,
            num_workers=40, pin_memory=True)
    accuracy = test_accuracy(classifier, transport_loader)
    return accuracy