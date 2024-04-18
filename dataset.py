from os.path import splitext
from os import listdir
import numpy as np
import os
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision.transforms import functional as TF
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import imageio.v2 as imageio
import rasterio


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='', tif=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.tif = tif
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, cv2_img, scale):
        w = cv2_img.shape[0]
        h = cv2_img.shape[1]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        img_nd = np.array(cv2_img)

        if len(cv2_img.shape) > 2:

            img_nd = img_nd.transpose((2, 0, 1))
            if img_nd.max() > 1:
                img_nd = img_nd / 255

        return img_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask = TIFF.open(mask_file[0], mode='r').read_image()
        img = TIFF.open(img_file[0], mode='r').read_image()

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        # print(mask//80)

        return {
            'image': torch.from_numpy(img).type(torch.float32),
            'mask': torch.from_numpy(mask//80).type(torch.uint8)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, tif=False):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask', tif=tif)




class RasterTilesDataset(Dataset):
    def __init__(self, dem_dir, so_dir, transform=None):
        """
        Custom dataset to load DEM and SO tiles.

        :param dem_dir: Directory where DEM tiles are stored.
        :param so_dir: Directory where SO tiles are stored.
        :param transform: Optional transform to be applied on a sample.
        """
        self.dem_dir = dem_dir
        self.so_dir = so_dir
        self.transform = transform

        # Extracting unique identifiers (coordinates) from DEM filenames
        self.tile_identifiers = [f.split('_')[2:4] for f in os.listdir(dem_dir) if 'dem_tile' in f]

    def __len__(self):
        return len(self.tile_identifiers)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tile_id = self.tile_identifiers[idx]
        dem_file = os.path.join(self.dem_dir, f'dem_tile_{tile_id[0]}_{tile_id[1]}')
        so_file = os.path.join(self.so_dir, f'so_tile_{tile_id[0]}_{tile_id[1]}')

        dem_image = Image.open(dem_file)
        so_image = Image.open(so_file)

        dem_array = np.array(dem_image)
        so_array = np.array(so_image)

        sample = {'DEM': dem_image, 'SO': so_image}

        if self.transform:
            sample = self.transform(sample)

        return sample




class RasterTransform:
    """
    A custom transform class for raster data.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        dem, so = sample['DEM'], sample['SO']

        # Random horizontal flipping
        # if torch.rand(1) > 0.5:
        #     dem = TF.hflip(dem)
        #     so = TF.hflip(so)

        # # Random vertical flipping
        # if torch.rand(1) > 0.5:
        #     dem = TF.vflip(dem)
        #     so = TF.vflip(so)

        # Convert numpy arrays to tensors
        dem = TF.to_tensor(dem)
        so = TF.to_tensor(so)

        dem = TF.normalize(dem, 318.90567, 16.467052)

        so = so.long()

        return {'DEM': dem, 'SO': so.squeeze()}


class RGB_RasterTilesDataset(Dataset):
    def __init__(self, dem_dir, so_dir, rgb_dir, transform=None):
        """
        Custom dataset to load DEM, SO, and RGB tiles.

        :param dem_dir: Directory where DEM tiles are stored.
        :param so_dir: Directory where SO tiles are stored.
        :param rgb_dir: Directory where RGB tiles are stored.
        :param transform: Optional transform to be applied on a sample.
        """
        self.dem_dir = dem_dir
        self.so_dir = so_dir
        self.rgb_dir = rgb_dir
        self.transform = transform

        # self.filenames = [f for f in os.listdir(dem_dir) if os.path.isfile(os.path.join(dem_dir, f))]
        self.tile_identifiers = [f.split('_')[-1] for f in os.listdir(dem_dir) if 'dem_tile' in f]

    def __len__(self):
        return len(self.tile_identifiers)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tile_id = self.tile_identifiers[idx]
        dem_file = os.path.join(self.dem_dir, f'dem_tile_{tile_id}')
        so_file = os.path.join(self.so_dir, f'so_tile_{tile_id}')

        # dem_file = os.path.join(self.dem_dir, self.filenames[idx])
        # so_file = os.path.join(self.so_dir, self.filenames[idx])
        # Assuming RGB tiles follow a similar naming convention
        rgb_files = [os.path.join(self.rgb_dir, f'rgb{k}_tile_{tile_id}') for k in range(6)]

        dem_image = Image.open(dem_file)
        so_image = Image.open(so_file)
        rgb_images = [imageio.imread(file) for file in rgb_files]

        dem_array = np.array(dem_image)
        so_array = np.array(so_image)
        rgb_arrays = [np.array(image).transpose(2,0,1)/255 for image in rgb_images]

        sample = {'DEM': dem_array, 'SO': so_array, 'RGB': rgb_arrays}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class RGB_RasterTilesDataset_Geo(Dataset):
    def __init__(self, dem_dir, so_dir, rgb_dir, transform=None):
        self.dem_dir = dem_dir
        self.so_dir = so_dir
        self.rgb_dir = rgb_dir
        self.transform = transform
        # Assume all DEM, SO, and RGB files share the same tile identifiers
        self.tile_identifiers = [f.split('_')[-1].split('.')[0] for f in os.listdir(dem_dir) if 'dem_tile' in f]

    def __len__(self):
        return len(self.tile_identifiers)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tile_id = self.tile_identifiers[idx]
        dem_file = os.path.join(self.dem_dir, f'dem_tile_{tile_id}.tif')
        so_file = os.path.join(self.so_dir, f'dem_tile_{tile_id}.tif')
        rgb_files = [os.path.join(self.rgb_dir, f'rgb{k}_tile_{tile_id}.tif') for k in range(6)]

        # Prepare sample dictionary
        sample = {}

        # Read DEM file and extract the transform
        with rasterio.open(dem_file) as src:
            dem_image = src.read(1)  # Read the first band
            dem_transform = src.transform
            sample['DEM'] = dem_image
            sample['DEM_transform'] = dem_transform

        # Read SO file and extract the transform
        with rasterio.open(so_file) as src:
            so_image = src.read(1)
            so_transform = src.transform
            sample['SO'] = so_image
            sample['SO_transform'] = so_transform

        # Read RGB files and extract their transforms
        rgb_images = []
        rgb_transforms = []
        for file in rgb_files:
            with rasterio.open(file) as src:
                rgb_images.append(src.read([1, 2, 3]))  # Read the RGB bands
                rgb_transforms.append(src.transform)
        sample['RGB'] = rgb_images
        sample['RGB_transforms'] = rgb_transforms

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class RGB_RasterTransform:
    """
    A custom transform class for raster data.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        dem, so, rgb = sample['DEM'], sample['SO'], sample['RGB']

        # Random horizontal flipping
        # if torch.rand(1) > 0.5:
        #     dem = TF.hflip(dem)
        #     so = TF.hflip(so)

        # # Random vertical flipping
        # if torch.rand(1) > 0.5:
        #     dem = TF.vflip(dem)
        #     so = TF.vflip(so)

        # Convert numpy arrays to tensors
        dem = TF.to_tensor(dem)
        so = TF.to_tensor(so)
        rgb_images = [TF.to_tensor(image) for image in rgb]

        dem = TF.normalize(dem, 318.90567, 16.467052)

        so = so.long()

        return {'DEM': dem, 'SO': so.squeeze(), 'RGB': rgb}
    

class RGB_RasterTransform_Geo:
    """
    A custom transform class for raster data.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        dem, so, rgb = sample['DEM'], sample['SO'], sample['RGB']
        dem_meta, so_meta, rgb_meta = sample['DEM_transform'], sample['SO_transform'], sample['RGB_transforms']

        # Random horizontal flipping
        # if torch.rand(1) > 0.5:
        #     dem = TF.hflip(dem)
        #     so = TF.hflip(so)

        # # Random vertical flipping
        # if torch.rand(1) > 0.5:
        #     dem = TF.vflip(dem)
        #     so = TF.vflip(so)

        # Convert numpy arrays to tensors
        dem = TF.to_tensor(dem)
        so = TF.to_tensor(so)
        rgb_images = [TF.to_tensor(image) for image in rgb]
        float_rgb_images = [image.float() for image in rgb_images]
        # rgb_images = rgb_images.float()

        dem = TF.normalize(dem, 318.90567, 16.467052)

        so = so.long()

        return {'DEM': dem, 'SO': so.squeeze(), 'RGB': float_rgb_images,
                'DEM_transform' : dem_meta, 'SO_transform' : so_meta, 'RGB_transforms' : rgb_meta}