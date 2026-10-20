"""
Loads the generated pattern images for PyTorch.
"""
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from groups import GROUP_NAMES, NAME_TO_IDX

IMG_SIZE = 128


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return preprocess(img)


class PatternDataset(Dataset):
    def __init__(self, root, split):
        self.samples = []
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"{split_dir} not found. Run generate_data.py first.")

        for name in GROUP_NAMES:
            folder = os.path.join(split_dir, name)
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(folder, fname), NAME_TO_IDX[name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return load_image(path), label


def get_loaders(root="data/synthetic", batch_size=32):
    train = PatternDataset(root, "train")
    val = PatternDataset(root, "val")
    test = PatternDataset(root, "test")
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()
    images, labels = next(iter(train_loader))
    print("batch of images:", tuple(images.shape))
    print("labels:", labels[:8].tolist())
    print("train/val/test sizes:", len(train_loader.dataset), len(val_loader.dataset),
          len(test_loader.dataset))

