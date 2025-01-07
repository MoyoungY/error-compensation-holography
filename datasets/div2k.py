import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import utils.utils as utils
from skimage.transform import resize


class DIV2KDataset(Dataset):
    def __init__(self, data_path, channel=None, image_res=(1080, 1920),
                 homography_res=(880, 1600), transform=None, split='train'):
        self.data_path = data_path
        self.channel = channel
        self.image_res = image_res
        self.homography_res = homography_res

        self.split = split
        if self.split == 'train':
            self.transform = transforms.Compose([
                # transforms.Resize(size=homography_res),
                transforms.RandomHorizontalFlip(), 
                transforms.RandomVerticalFlip(),  
                # transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize(size=homography_res)
                # transforms.ToTensor()
            ])

        self.im_names = sorted(os.listdir(data_path))

    def __len__(self):
        return len(self.im_names)

    def load_image(self, im_path):
        im = np.array(Image.open(im_path))

        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)  # augment channels for gray images

        if self.channel is None:
            im = im[..., :3]  # remove alpha channel, if any
        else:
            # select channel while keeping dims
            im = im[..., self.channel, np.newaxis]

        im = utils.im2float(im, dtype=np.float32)  # convert to double, max 1

        # linearize intensity and convert to amplitude
        low_val = im <= 0.04045
        im[low_val] = 25 / 323 * im[low_val]
        im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11)
                                       / 211) ** (12 / 5)
        im = np.sqrt(im)  # to amplitude

        # move channel dim to torch convention
        im = np.transpose(im, axes=(2, 0, 1))

        # normalize resolution
        im = resize_keep_aspect(im, self.homography_res)
        im = pad_crop_to_res(im, self.image_res)
        return im


    def __getitem__(self, idx):
        im_path = os.path.join(self.data_path, self.im_names[idx])
        im = self.load_image(im_path)
        im = torch.from_numpy(im)

        if self.transform:
            im = self.transform(im)

        return im


def resize_keep_aspect(image, target_res, pad=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # finds the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res, pytorch=False)

    # switch to numpy channel dim convention, resize, switch back
    image = np.transpose(image, axes=(1, 2, 0))
    image = resize(image, target_res, mode='reflect')
    return np.transpose(image, axes=(2, 0, 1))


def pad_crop_to_res(image, target_res):
    """Pads with 0 and crops as needed to force image to be target_res

    image: an array with dims [..., channel, height, width]
    target_res: [height, width]
    """
    return utils.crop_image(utils.pad_image(image,
                                            target_res, pytorch=False),
                            target_res, pytorch=False)

if __name__ == "__main__":
    dataset = DIV2KDataset("../data/DIV2K_valid_HR", channel=1)
    dataset.load_image("../data/DIV2K_valid_HR/0801.png")