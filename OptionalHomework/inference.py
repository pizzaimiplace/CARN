import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from pathlib import Path
from timed_decorator.simple_timed import timed

class SimpleTransformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3),  # Input: 3x32x32 -> 16x30x30
            nn.ReLU(),
            nn.Conv2d(16, 1, 3),  # Output: 1x28x28
        )

    def forward(self, x):
        return self.net(x)


def get_test_images(data_path="./data"):
    initial_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    dataset = CIFAR10(root=data_path, train=False, transform=initial_transforms, download=True)
    return torch.stack([img for img, _ in dataset])


@timed(return_time=True, use_seconds=True, stdout=False)
def benchmark_sequential_cpu(images):
    transforms = v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])
    for img in images:
        _ = transforms(img)


@timed(return_time=True, use_seconds=True, stdout=False)
@torch.no_grad()
def benchmark_model(images, model, device, batch_size):
    model.eval()
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for (x,) in loader:
        _ = model(x.to(device))


def run_benchmarks(model, images):
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    batch_sizes = [1, 16, 64, 128, 256]

    print(f"{'Device':<10} | {'Batch Size':<10} | {'Sequential (s)':<15} | {'Model (s)':<15} | {'Status'}")
    print("-" * 75)

    _, t_seq = benchmark_sequential_cpu(images)

    for device in devices:
        model.to(device)
        for bs in batch_sizes:
            _, t_model = benchmark_model(images, model, device, bs)
            status = "FASTER" if t_model < t_seq else "Slower"

            print(f"{str(device):<10} | {bs:<10} | {t_seq:<15.4f} | {t_model:<15.4f} | {status}")


if __name__ == "__main__":
    test_images = get_test_images()
    model = SimpleTransformNet()
    weights_path = Path("weights.pth")

    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"Loaded weights from {weights_path}")
    else:
        print("Error: weights.pth not found")
        exit()

    run_benchmarks(model, test_images)