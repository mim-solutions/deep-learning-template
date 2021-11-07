import torchvision.transforms as transforms
from . import transforms as our_transforms


class DefaultTransformFactory:

    def __init__(self, shape):
        self.shape = shape

    def __call__(self):
        return transforms.Compose([
            transforms.ToTensor(),
            our_transforms.Ensure3ChannelTransform(),
            transforms.Resize(self.shape)
        ])


class DefaultAugmentationFactory:

    def __call__(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2)
        ])
