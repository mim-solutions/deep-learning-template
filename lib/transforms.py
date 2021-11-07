import torch


class Ensure3ChannelTransform:
    # TODO - this class is hacky
    # Some images have more then 3 channels (thermal channel etc.)
    # or 1 channel (greyscale)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img[:3]
        if img.shape[0] < 3:
            img = img[0].unsqueeze(0).expand(3, -1, -1)

        return img
