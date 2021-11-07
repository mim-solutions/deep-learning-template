import torch
import skimage
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path
import os


class FakeVideoFromImages(Dataset):

    def __init__(self, image_loader, video_len=5):
        self.videos = []
        self.targets = []
        self.movies = defaultdict(list)

        for data, target in image_loader:
            self.movies[target].append(data)

        # cycle to the beginning for the last image
        for target in self.movies.keys():
            self.movies[target] += self.movies[target][:video_len - 1]

        for target, movie in self.movies.items():
            for i in range(len(movie) - video_len + 1):
                self.videos.append(torch.stack(movie[i:i + video_len]))
                self.targets.append(target)

    def __getitem__(self, key):
        return (self.videos[key], self.targets[key])

    def __len__(self):
        return len(self.videos)


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, file_names, name_to_label, transform=None):
        self.file_names = file_names
        self.name_to_label = name_to_label
        self.transform = transform

    def __getitem__(self, key):
        file_name = self.file_names[key]
        name = Path(file_name).stem
        # data = torchvision.io.read_image(file_name) # does not read certain file formats
        data = skimage.io.imread(file_name)

        if self.transform:
            data = self.transform(data)

        return data, self.name_to_label[name]

    def __len__(self):
        return len(self.file_names)


class Subset(torch.utils.data.Dataset):

    def __init__(self, dataset, targets, per_target, random=False):
        """
            Picks subset of the dataset. 
        Arguments
            random: if False, always picks the same indices for given parameters.
            targets are converted to labels: [3,5,1] -> [0,1,2]
        """
        self.dataset = dataset
        self.indices = []
        self.target_to_label = {t: i for i, t in enumerate(targets)}

        iterable = range(len(dataset))
        if random:
            iterable = torch.utils.data.SubsetRandomSampler(iterable)

        num_per_target = {t: 0 for t in targets}
        for i in iterable:
            _, target = dataset[i]
            if target in targets and num_per_target[target] < per_target:
                num_per_target[target] += 1
                self.indices.append(i)
            if len(self.indices) == len(targets) * per_target:
                break

    def __getitem__(self, idx):
        data, target = self.dataset[self.indices[idx]]
        return data, self.target_to_label[target]

    def __len__(self):
        return len(self.indices)


class ICubDataset(torch.utils.data.Dataset):

    label_map = ['book', 'cellphone', 'mouse', 'pencilcase', 'ringbinder']

    # example image path:
    # PosixPath('/dysk1/approx/videoresearch/icubworld/part1_cropped/cellphone/cellphone7/SCALE/day3/left/00001877.jpg')

    def __init__(self, data_root_path: str = "/dysk1/approx/videoresearch/icubworld/part1_cropped",
                 transform=None, test: bool = False, video_len: int = 5):
        self.transform = transform
        if video_len == 0:
            self.return_images = True
            self.video_len = 1
        else:
            self.return_images = False
            self.video_len = video_len

        data_root_path = Path(data_root_path)
        # get all pictures recursively
        paths = list(data_root_path.rglob('*.jpg'))

        videos = defaultdict(lambda: [])
        splits = [os.path.split(p) for p in paths]  # dir_name and file_name
        for dir_name, file_name in splits:
            if not file_name.startswith('.'):  # skip hidden files
                videos[dir_name].append(file_name)

        for dir_name, vid in videos.items():
            # make sure frames are sorted
            videos[dir_name] = sorted(vid, key=lambda frame_name: int(Path(frame_name).stem))

        labeled_data = defaultdict(dict)

        for dir_name, vid in videos.items():
            rel_path = os.path.relpath(dir_name, start=data_root_path)
            label, item = rel_path.split('/')[:2]

            if test == item.endswith('10'):  # only last item for test
                labeled_data[label][dir_name] = vid

        self.data = []
        for label, vid_dict in labeled_data.items():
            for dir_name, vid in vid_dict.items():
                directory = Path(dir_name)
                for i in range(len(vid) - self.video_len + 1):
                    full_vid_names = [str(directory / p) for p in vid[i:i + self.video_len]]
                    datapoint = (self.label_map.index(label), full_vid_names)
                    self.data.append(datapoint)

    def __getitem__(self, key):
        label, vid = self.data[key]

        data = []
        for frame_name in vid:
            frame = skimage.io.imread(frame_name)
            if self.transform:
                frame = self.transform(frame)
            data.append(frame)

        if self.return_images:
            data = data[0]  # video_len == 1
        else:
            data = torch.stack(data)

        return data, label

    def __len__(self):
        return len(self.data)


def video_to_images_transform_fn(model, data, target, normalization):
    t_data = data.reshape(data.shape[0] * data.shape[1], 3, 224, 224)

    t_target = torch.repeat_interleave(target, repeats=data.shape[1], dim=0)

    return t_data, t_target


class ImagesFromVideo(torch.utils.data.Dataset):

    def __init__(self, video_dataset):
        self.data = []
        for video, label in iter(video_dataset):
            for frame in video:
                self.data.append((frame, label))

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


class VideoFromImages(torch.utils.data.Dataset):

    def __init__(self, image_dataset, video_len):
        self.data = []
        for image, label in iter(image_dataset):
            self.data.append((torch.stack(video_len * [image]), label))

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)
