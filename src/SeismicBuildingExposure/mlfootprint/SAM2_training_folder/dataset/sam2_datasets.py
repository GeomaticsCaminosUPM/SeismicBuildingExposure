# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

from typing import Callable, Iterable, List, Optional, Sequence

import torch

from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Subset

from torch.utils.data.distributed import DistributedSampler

import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import os
import matplotlib.pyplot as plt

class MixedDataLoader:
    def __init__(self, dataloaders: List[DataLoader], mixing_prob: torch.FloatTensor):
        """
        Args:
            dataloaders (List[DataLoader]): List of DataLoaders to be mixed.
            mixing_prob (torch.FloatTensor): Probability of each dataloader to be sampled from

        """
        assert len(dataloaders) == mixing_prob.shape[0]
        self.dataloaders = dataloaders
        self.mixing_prob = mixing_prob
        # Iterator state
        self._iter_dls = None
        self._iter_mixing_prob = None
        self.random_generator = torch.Generator()

    def __len__(self):
        return sum([len(d) for d in self.dataloaders])

    def __iter__(self):
        # Synchronize dataloader seeds
        self.random_generator.manual_seed(42)
        self._iter_dls = [iter(loader) for loader in self.dataloaders]
        self._iter_mixing_prob = self.mixing_prob.clone()
        return self

    def __next__(self):
        """
        Sample a dataloader to sample from based on mixing probabilities. If one of the dataloaders is exhausted, we continue sampling from the other loaders until all are exhausted.
        """
        if self._iter_dls is None:
            raise TypeError(f"{type(self).__name__} object is not an iterator")

        while self._iter_mixing_prob.any():  # at least one D-Loader with non-zero prob.
            dataset_idx = self._iter_mixing_prob.multinomial(
                1, generator=self.random_generator
            ).item()
            try:
                item = next(self._iter_dls[dataset_idx])
                return item
            except StopIteration:
                # No more iterations for this dataset, set it's mixing probability to zero and try again.
                self._iter_mixing_prob[dataset_idx] = 0
            except Exception as e:
                # log and raise any other unexpected error.
                logging.error(e)
                raise e

        # Exhausted all iterators
        raise StopIteration


class TorchTrainMixedDataset:
    def __init__(
        self,
        datasets: List[Dataset],
        batch_sizes: List[int],
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        phases_per_epoch: int = 1,
        dataset_prob: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            datasets (List[Dataset]): List of Datasets to be mixed.
            batch_sizes (List[int]): Batch sizes for each dataset in the list.
            num_workers (int): Number of workers per dataloader.
            shuffle (bool): Whether or not to shuffle data.
            pin_memory (bool): If True, use pinned memory when loading tensors from disk.
            drop_last (bool): Whether or not to drop the last batch of data.
            collate_fn (Callable): Function to merge a list of samples into a mini-batch.
            worker_init_fn (Callable): Function to init each dataloader worker.
            phases_per_epoch (int): Number of phases per epoch.
            dataset_prob (List[float]): Probability of choosing the dataloader to sample from. Should sum to 1.0
        """

        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        assert len(self.datasets) > 0
        for dataset in self.datasets:
            assert not isinstance(dataset, IterableDataset), "Not supported"
            # `RepeatFactorWrapper` requires calling set_epoch first to get its length
            self._set_dataset_epoch(dataset, 0)
        self.phases_per_epoch = phases_per_epoch
        self.chunks = [None] * len(datasets)
        if dataset_prob is None:
            # If not provided, assign each dataset a probability proportional to its length.
            dataset_lens = [
                (math.floor(len(d) / bs) if drop_last else math.ceil(len(d) / bs))
                for d, bs in zip(datasets, batch_sizes)
            ]
            total_len = sum(dataset_lens)
            dataset_prob = torch.tensor([d_len / total_len for d_len in dataset_lens])
        else:
            assert len(dataset_prob) == len(datasets)
            dataset_prob = torch.tensor(dataset_prob)

        logging.info(f"Dataset mixing probabilities: {dataset_prob.tolist()}")
        assert dataset_prob.sum().item() == 1.0, "Probabilities should sum to 1.0"
        self.dataset_prob = dataset_prob

    def _set_dataset_epoch(self, dataset, epoch: int) -> None:
        if hasattr(dataset, "epoch"):
            dataset.epoch = epoch
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def get_loader(self, epoch) -> Iterable:
        dataloaders = []
        for d_idx, (dataset, batch_size) in enumerate(
            zip(self.datasets, self.batch_sizes)
        ):
            if self.phases_per_epoch > 1:
                # Major epoch that looops over entire dataset
                # len(main_epoch) == phases_per_epoch * len(epoch)
                main_epoch = epoch // self.phases_per_epoch

                # Phase with in the main epoch
                local_phase = epoch % self.phases_per_epoch

                # Start of new data-epoch or job is resumed after preemtion.
                if local_phase == 0 or self.chunks[d_idx] is None:
                    # set seed for dataset epoch
                    # If using RepeatFactorWrapper, this step currectly re-samples indices before chunking.
                    self._set_dataset_epoch(dataset, main_epoch)

                    # Separate random generator for subset sampling
                    g = torch.Generator()
                    g.manual_seed(main_epoch)
                    self.chunks[d_idx] = torch.chunk(
                        torch.randperm(len(dataset), generator=g),
                        self.phases_per_epoch,
                    )

                dataset = Subset(dataset, self.chunks[d_idx][local_phase])
            else:
                self._set_dataset_epoch(dataset, epoch)

            sampler = DistributedSampler(dataset, shuffle=self.shuffle)
            sampler.set_epoch(epoch)

            batch_sampler = BatchSampler(sampler, batch_size, drop_last=self.drop_last)
            dataloaders.append(
                DataLoader(
                    dataset,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    batch_sampler=batch_sampler,
                    collate_fn=self.collate_fn,
                    worker_init_fn=self.worker_init_fn,
                )
            )
        return MixedDataLoader(dataloaders, self.dataset_prob)


class CocoSAM2Dataset(Dataset):
    def __init__(self, annotation_file, n_classes, root=None, transform=None):
        if root is None:
            root = os.path.dirname(annotation_file)

        self.img_folder = os.path.normpath(root)
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())

        from transformers import Mask2FormerImageProcessor

        processor = Mask2FormerImageProcessor(
            reduce_labels=True,
            ignore_index=0,
            do_resize=False,
            do_rescale=False,
            do_normalize=True
        )

        processor.mean = [0.485, 0.456, 0.406]
        processor.std = [0.229, 0.224, 0.225]

        self.processor = processor
        
        self.transform = transform
        self.n_classes = n_classes

    def __len__(self):
        return len(self.image_ids)

    def image_id(self,idx):
        return self.image_ids[idx]
        
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        image = np.array(Image.open(img_path).convert("RGB"))

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        height, width = img_info['height'], img_info['width']
        
        masks = []
        class_ids = [0]

        for ann in anns:
            rle = ann['segmentation']
            if isinstance(rle, list):
                rle = mask_utils.frPyObjects(rle, height, width)
                rle = mask_utils.merge(rle)

            mask = mask_utils.decode(rle)
            # Ensure we get only binary mask (some RLEs are multi-channel)
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            masks.append(mask)#*ann['id'])
            class_ids.append(ann['category_id'])

        # If no annotations, return dummy
        if len(masks) == 0:
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.uint8)
            return inputs

        # Stack masks into a single array: (num_instances, H, W)
        masks = np.stack(masks,axis=0)
        masks = binary_to_logit_mask(masks)

        # Optional: apply transforms (e.g. Albumentations or torchvision)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=masks)
            image, masks = transformed['image'], transformed['mask']
            image[image > 255] = 255
            image[image < 0] = 0
        else:
            None

        pil_image = Image.fromarray(image)

        # Create instance_id to semantic_id mapping
        inst2class = {i: class_ids[i] for i in range(len(class_ids))}

        # Process inputs
        inputs = self.processor([image], [masks], instance_id_to_semantic_id=inst2class, return_tensors="pt")
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        class_labels = inputs["class_labels"]
        mask_labels = inputs["mask_labels"]

        # Get binary masks and points

        inds = np.arange(mask_labels.shape[0])
        points= []
        #masks = []
        for ind in inds:
            mask = mask_labels[ind, :, :].to(torch.uint8)  # Convert to uint8
            coords = (mask > 0).nonzero(as_tuple=False)    # Get all (y, x) where mask > 0
            rand_idx = torch.randint(0, coords.size(0), (1,))  # Pick a random index
            yx = coords[rand_idx].squeeze(0)               # Shape: (2,)
            points.append([[yx[1].item(), yx[0].item()]])  # Append as [x, y]
                        
        inputs = {
            'image':pil_image,
            'pixel_mask':inputs["pixel_mask"],
            'mask_labels':inputs["mask_labels"],#np.array(masks),
            'points':torch.Tensor(np.array(points)),
            'class_labels':class_labels
        }
        return inputs