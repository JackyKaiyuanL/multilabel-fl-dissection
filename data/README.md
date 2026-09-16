# Datasets (not included)

All datasets used in the paper are third-party, published datasets and are **not
redistributed** here. Download them from their owners into this `data/` directory
(the code reads `./data` relative to the repository root):

| Dataset | Where it comes from | Expected location |
|---|---|---|
| Pascal VOC 2007 (trainval + test) | downloaded automatically by `torchvision.datasets.VOCDetection(download=True)` on first use | `data/VOCdevkit/VOC2007/` |
| MS-COCO 2017 `val2017` images + `annotations/instances_val2017.json` | https://cocodataset.org/#download | `data/coco/val2017/`, `data/coco/annotations/` |
| CIFAR-10 / CIFAR-100 | downloaded automatically by torchvision | `data/cifar-10-batches-py/`, `data/cifar-100-python/` |
| MNIST | downloaded automatically by torchvision | `data/MNIST/` |
| FLAIR (Song, Granqvist & Talwar, NeurIPS D&B 2022) | Apple's release: https://github.com/apple/ml-flair (labels_and_metadata.json + the `small_images-XX.tar.gz` shards) | `data/flair/labels_and_metadata.json`, `data/flair/small_images/small_images/*.jpg` |

Please respect each dataset's own license and terms of use. FLAIR in particular is
released under Apple's license and Flickr's terms; the code here only reads the
official files and never redistributes images.
