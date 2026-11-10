import torch
import torchvision
import torchvision.transforms as transforms
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
train_data = torchvision.datasets.KMNIST("./data", train=True, download=True,transform=transform)
test_data = torchvision.datasets.KMNIST("./data", train=False, download=True,transform=transform)

X_train = torch.stack([image.flatten() for image, label in train_data]).to(device)
X_test = torch.stack([image.flatten() for image, label in test_data]).to(device)
y_train = torch.tensor([label for image, label in train_data], dtype=torch.long, device=device)
y_test = torch.tensor([label for image, label in test_data], dtype=torch.long, device=device)

pca_start = time.time()
# Calculate training mean
mean = X_train.mean(dim=0)
# Center training data
X_centered = X_train - mean
# Calculate covariance matrix
covariance = (
    X_centered.T @ X_centered
) / (X_centered.shape[0] - 1)
# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

# Sort from largest to smallest
indices = torch.argsort(eigenvalues,descending=True)
eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:, indices]

# Calculate explained variance
explained_variance = (eigenvalues / eigenvalues.sum())
cumulative_variance = torch.cumsum(explained_variance, dim=0)

# Find components explaining 95% variance
C = (torch.searchsorted(cumulative_variance,torch.tensor(0.95, device=device)).item() + 1)

print("Number of PCA components:", C)
print("Explained variance:", cumulative_variance[C-1].item() * 100,"%")
# Keep first C eigenvectors
components = eigenvectors[:, :C]
# Project training and testing data
X_train_pca = X_centered @ components
X_test_pca = (X_test - mean) @ components
pca_time = time.time() - pca_start

print("Training PCA shape:", X_train_pca.shape)
print("Testing PCA shape:", X_test_pca.shape)
print("PCA time:", pca_time, "seconds")

# Save PCA results
torch.save({
    "mean": mean,
    "components": components,
    "X_train_pca": X_train_pca,
    "X_test_pca": X_test_pca,
    "y_train": y_train,
    "y_test": y_test,
    "C": C
}, "pca_results.pth")
