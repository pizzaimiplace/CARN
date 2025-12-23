import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torchvision.utils import save_image
from pathlib import Path


class CustomDataset(Dataset):
    def __init__(self, data_path="./data", train=True, cache=True):
        initial_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        dataset = CIFAR10(root=data_path, train=train, transform=initial_transforms, download=True)
        self.images = [img for img, _ in dataset]

        self.cache = cache
        self.transforms = v2.Compose([
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            v2.functional.hflip,
            v2.functional.vflip,
        ])

        if cache:
            print(f"Caching {'train' if train else 'test'} transformations...")
            self.labels = [self.transforms(x) for x in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.cache:
            return self.images[idx], self.labels[idx]
        return self.images[idx], self.transforms(self.images[idx])


class SimpleTransformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0),  # 32x32 -> 30x30
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=0),  # 30x30 -> 28x28
        )

    def forward(self, x):
        return self.net(x)


def train(device, epochs=50, batch_size=128, patience=5):
    dataset = CustomDataset(train=True, cache=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter(log_dir="logs")

    model = SimpleTransformNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    patience_ctr = 0

    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch + 1:02d} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_ctr = 0
            torch.save(model.state_dict(), "weights.pth")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping triggered")
                break

    writer.close()
    return model

def save_visual_results(model, device, n=5):
    model.eval()
    out_dir = Path("examples")
    out_dir.mkdir(exist_ok=True)

    test_dataset = CustomDataset(train=False, cache=False)

    print(f"Saving {n} comparison images to /examples...")
    with torch.no_grad():
        for i in range(n):
            img_in, img_gt = test_dataset[i]
            img_pred = model(img_in.unsqueeze(0).to(device)).cpu().squeeze(0)

            save_image(img_gt, out_dir / f"sample_{i}_ground_truth.png")
            save_image(img_pred, out_dir / f"sample_{i}_prediction.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train(device)

    save_visual_results(model, device)

    print("Training complete. Weights saved to 'weights.pth'.")

if __name__ == "__main__":
    main()