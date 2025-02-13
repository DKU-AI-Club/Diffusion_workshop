import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_transformed_dataset(img_size=32, batch_size=128) -> DataLoader:
    """Load and transform the CIFAR10 dataset."""
    train_data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ToTensor(),  # Scale data to range [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize data to range [-1, 1]
    ])
    test_data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load training and test datasets
    train_dataset = torchvision.datasets.CIFAR10(root="./datasets", 
                                                train=True,
                                                download=False,
                                                transform=train_data_transform)
    
    test_dataset = torchvision.datasets.CIFAR10(root="./datasets",
                                               train=False, 
                                               download=False,
                                               transform=test_data_transform)

    # Create DataLoader
    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)
    
    test_loader = DataLoader(test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           drop_last=True)
    
    return train_loader, test_loader


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # Scale data from [-1, 1] to [0, 1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # Change channel order from CHW to HWC
        transforms.Lambda(lambda t: t * 255.),  # Scale data to range [0, 255]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  # Convert to uint8 type
        transforms.ToPILImage(),  # Convert to PIL image format
    ])

    # If the image is a batch, take the first image
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)


def show_tensor_images_batch(images):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # Scale data from [-1, 1] to [0, 1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # Change channel order from CHW to HWC
        transforms.Lambda(lambda t: t * 255.),  # Scale data to range [0, 255]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  # Convert to uint8 type
        transforms.ToPILImage(),  # Convert to PIL image format
    ])

    batch_size = images.shape[0]  # Get batch size
    cols = 8  # Set number of images per row
    rows = batch_size // cols  # Calculate number of rows
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i in range(batch_size):
        image = reverse_transforms(images[i])  # Process individual image
        ax = axes[i // cols, i % cols]  # Compute row and column index
        ax.imshow(image)
        ax.axis('off')  # Hide axes
    plt.show()


if __name__ == "__main__":
    train_loader, test_loader = load_transformed_dataset()
    # for i, batch in enumerate(train_loader):
    #     image, _ = next(iter(train_loader))  # Get a batch of images
    #     show_tensor_images_batch(image)  # Show all images in the batch
        
    image, _ = next(iter(train_loader))  # Get a batch of images
    plt.imshow(show_tensor_image(image))
    plt.show()
