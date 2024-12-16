import os
import random

import h5py
import numpy as np
import torch
import ujson as json
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from timm.data.transforms_factory import create_transform
from torchvision.transforms import transforms


class HOIdataset_h5py(Dataset):
    def __init__(
        self,
        data_path: str,
        annotation_path: str,
        dataset_type: str,
        img_size: int | None,
    ):

        self.data_files = [os.path.join(annotation_path, dataset_type)]
        if dataset_type:
            self.data_files = [f for f in self.data_files if dataset_type in f]
        self.file_sizes = []
        for file in self.data_files:
            with open(file) as f:
                file_hoi = json.load(f)
                self.file_sizes.append(len(file_hoi))

        self.idx_ranges = np.cumsum(self.file_sizes)

        if img_size:
            self.transforms = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size)),
                ]
            )
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

        self.data_path = data_path

    def __len__(self):
        return int(np.sum(self.file_sizes))

    def __getitem__(self, item):
        idx = 0
        for i in range(len(self.idx_ranges)):
            if item < self.idx_ranges[i]:
                idx = i
                break

        file = self.data_files[idx]
        with open(file) as f:
            files_hoi = json.load(f)

        if idx == 0:
            file_hoi = files_hoi[item]
        else:
            file_hoi = files_hoi[item - self.idx_ranges[idx - 1]]

        context = file_hoi[0] + file_hoi[1]
        context = [c["im_path"] for c in context]

        answers = []
        # adding random image as answer
        answers.append(context.pop(random.randint(0, 6)))
        answers.append(context.pop(random.randint(6, 12)))

        # random answer flip
        if random.uniform(0, 1) <= 0.5:
            target = np.asarray(0)
        else:
            target = np.asarray(1)
            answers = answers[::-1]

        images = context + answers
        images_data = []
        for im in images:
            if im.startswith("."):
                im = im[2:]
            folder, im_path = im.split("/", 1)
            with h5py.File(
                os.path.join(self.data_path, "bongard_hoi_" + folder + ".hy")
            ) as f:
                images_data.append(np.array(f[im]))
        img = [self.transforms(im) for im in images_data]
        img = torch.stack(img)

        return img, target


class LOGOdataset_h5py(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_files = os.path.join(data_path, f"bongard_logo_{dataset_type}.hy")
        with h5py.File(self.data_files) as f:
            self.file_length = len(f)
        if img_size:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((img_size, img_size))]
            )
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.file_length

    def __getitem__(self, item):
        with h5py.File(self.data_files) as f:
            images_grp1 = np.asarray(f[str(item)]["grp_1"])
            images_grp2 = np.asarray(f[str(item)]["grp_2"])
        ans_idx = random.randint(0, 6)
        answers = images_grp1[ans_idx]
        context = np.delete(images_grp1, ans_idx, axis=0)

        ans_idx = random.randint(0, 6)
        answers = np.concatenate(
            [np.expand_dims(answers, 0), np.expand_dims(images_grp2[ans_idx], 0)]
        )
        context = np.concatenate([context, np.delete(images_grp2, ans_idx, axis=0)])

        # random answer flip
        if random.uniform(0, 1) <= 0.5:
            target = np.asarray(0)
        else:
            target = np.asarray(1)
            answers = answers[::-1]

        images = np.concatenate([context, answers])

        images = [self.transforms(im) for im in images]
        img = torch.stack(images)

        return img, target


class LOGOSamplesDataset_h5py(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_files = os.path.join(data_path, f"bongard_logo_{dataset_type}.hy")
        with h5py.File(self.data_files) as f:
            self.file_length = len(f)
        if img_size:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((img_size, img_size))]
            )
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.file_length * 14

    def __getitem__(self, item):
        item = item // 14
        idx = item % 14
        grp = idx % 7
        grp_idx = (idx // 7) + 1
        with h5py.File(self.data_files) as f:
            images_grp = np.asarray(f[str(item)][f"grp_{grp_idx}"])[[grp]]
        image = [self.transforms(im) for im in images_grp]
        img = torch.stack(image)
        target = np.asarray(-1)

        return img, target


class CLEVRdataset_h5py(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_files = os.path.join(data_path, "clevr_" + dataset_type + ".hy")
        with h5py.File(self.data_files) as f:
            self.file_length = len(f)
        if img_size:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((img_size, img_size))]
            )
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.file_length

    def __getitem__(self, item):
        with h5py.File(self.data_files) as f:
            images = np.asarray(f[str(item)]["data"])
            target = np.asarray(f[str(item)]["target"][()])
        img = torch.stack([self.transforms(im) for im in images])

        return img, target


class MNSdataset_h5py(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_files = os.path.join(data_path, "mns_" + dataset_type + ".hy")
        with h5py.File(self.data_files) as f:
            self.file_length = len(f)
        if img_size:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((img_size, img_size))]
            )
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.file_length

    def __getitem__(self, item):
        with h5py.File(self.data_files) as f:
            images = np.asarray(f[str(item)]["data"])
            target = np.asarray(f[str(item)]["target"][()])

        img = torch.stack([self.transforms(im) for im in images])

        return img, target


class PGMdataset_h5py(Dataset):
    def __init__(
        self,
        data_path: str,
        regimes: list[str],
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_files = []
        self.file_sizes = []
        for regime in regimes:
            file = os.path.join(
                data_path, regime, "_".join(["pgm", regime, dataset_type]) + ".hy"
            )
            self.data_files.append(file)
            with h5py.File(file) as f:
                self.file_sizes.append(len(f))
        self.idx_ranges = np.cumsum(self.file_sizes)

        if img_size:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((img_size, img_size))]
            )
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return int(np.sum(self.file_sizes))

    def _return_file_idx(self, item):
        idx = 0
        for i in range(len(self.idx_ranges)):
            if item < self.idx_ranges[i]:
                idx = i
                break
        file = self.data_files[idx]
        if idx != 0:
            item = item - self.idx_ranges[idx - 1]
        return file, item

    def __getitem__(self, item):
        file, item = self._return_file_idx(item)
        with h5py.File(file) as f:
            images = np.asarray(f[str(item)]["data"])
            target = np.asarray(f[str(item)]["target"][()])

        img = torch.stack([self.transforms(im) for im in images])

        return img, target


class SVRTdataset_h5py(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_files = os.path.join(data_path, f"svrt_{dataset_type}.hy")
        with h5py.File(self.data_files) as f:
            self.file_length = len(f)
        if img_size:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((img_size, img_size))]
            )
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.file_length

    def __getitem__(self, item):
        with h5py.File(self.data_files) as f:
            images_grp1 = np.asarray(f[str(item)]["grp_1"])
            images_grp2 = np.asarray(f[str(item)]["grp_2"])
        ans_idx = random.randint(0, len(images_grp1) - 1)
        answers = images_grp1[ans_idx]
        context = np.delete(images_grp1, ans_idx, axis=0)

        ans_idx = random.randint(0, len(images_grp2) - 1)
        answers = np.concatenate(
            [np.expand_dims(answers, 0), np.expand_dims(images_grp2[ans_idx], 0)]
        )
        context = np.concatenate([context, np.delete(images_grp2, ans_idx, axis=0)])

        # random answer flip
        if random.uniform(0, 1) <= 0.5:
            target = np.asarray(0)
        else:
            target = np.asarray(1)
            answers = answers[::-1]

        images = np.concatenate([context, answers])

        images = [self.transforms(Image.fromarray(im).convert('RGB')) for im in images]
        img = torch.stack(images)

        return img, target


class SVRTdataset_h5py_vit(SVRTdataset_h5py):
    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        model_name: str,
    ):
        self.data_files = os.path.join(data_path, f"svrt_{dataset_type}.hy")
        with h5py.File(self.data_files) as f:
            self.file_length = len(f)

        model = timm.create_model(model_name)
        model_conf = timm.data.resolve_data_config({}, model=model)
        self.transforms = create_transform(**model_conf)


class LABCdataset_h5py(Dataset):
    def __init__(
        self,
        data_path: str,
        regimes: list[str],
        dataset_type: str,
        img_size: int | None,
    ):
        self.data_files = []
        self.file_sizes = []
        for regime in regimes:
            file = os.path.join(
                data_path, "_".join(["labc", regime, dataset_type]) + ".hy"
            )
            self.data_files.append(file)
            with h5py.File(file) as f:
                self.file_sizes.append(len(f))
        self.idx_ranges = np.cumsum(self.file_sizes)

        if img_size:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((img_size, img_size))]
            )
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return int(np.sum(self.file_sizes))

    def _return_file_idx(self, item):
        idx = 0
        for i in range(len(self.idx_ranges)):
            if item < self.idx_ranges[i]:
                idx = i
                break
        file = self.data_files[idx]
        if idx != 0:
            item = item - self.idx_ranges[idx - 1]
        return file, item

    def __getitem__(self, item):
        file, item = self._return_file_idx(item)
        with h5py.File(file) as f:
            images = np.asarray(f[str(item)]["data"])
            target = np.asarray(f[str(item)]["target"][()])

        img = torch.stack([self.transforms(im) for im in images])

        return img, target


def _test():

    # Example of usage
    print("Bongard HOI")
    train_dataset = HOIdataset_h5py(
        data_path=r"D:\mcs",
        annotation_path=r"D:\mcs\all_data\bongard_hoi\bongard_hoi_release",
        dataset_type="bongard_hoi_val_seen_obj_seen_act.json",
        img_size=80,
    )
    img, target = train_dataset[0]
    print(img.shape)
    print(target)

    print("Bongard LOGO")
    val_dataset_logo = LOGOdataset_h5py(
        data_path=r"D:\mcs", dataset_type="val", img_size=None
    )
    img, target = val_dataset_logo[0]
    print(img.shape)
    print(target)

    print("CLEVR")
    test_dataset_clevr = CLEVRdataset_h5py(
        data_path=r"D:\mcs", dataset_type="val", img_size=None
    )
    img, target = test_dataset_clevr[0]
    print(img.shape)
    print(target)

    print("MNS")
    test_dataset_mns = MNSdataset_h5py(
        data_path=r"D:\mcs", dataset_type="val", img_size=None
    )
    img, target = test_dataset_mns[0]
    print(img.shape)
    print(target)

    print("PGM")
    test_dataset_pgm = PGMdataset_h5py(
        data_path=r"D:\mcs",
        regimes=["extrapolation"],
        dataset_type="val",
        img_size=None,
    )
    img, target = test_dataset_pgm[0]
    print(img.shape)
    print(target)

    print("LABC")
    test_dataset_labc = LABCdataset_h5py(
        data_path=r"D:\mcs",
        regimes=["novel.target.domain.shape.color"],
        dataset_type="val",
        img_size=None,
    )
    img, target = test_dataset_labc[0]
    print(img.shape)
    print(target)

    print("SVRT")
    test_dataset_svrt = SVRTdataset_h5py(
        data_path=r"D:\mcs", dataset_type="val", img_size=None
    )
    img, target = test_dataset_svrt[0]
    print(img.shape)
    print(target)


if __name__ == "__main__":
    _test()
