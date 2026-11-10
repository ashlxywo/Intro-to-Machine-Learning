
import torch
import torchvision
import torchvision.transforms as transforms
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
train_data = torchvision.datasets.KMNIST("./data", train=True, download=True,transform=transform)
test_data = torchvision.datasets.KMNIST("./data", train=False, download=True,transform=transform)

# Convert images into feature vectors
X_train = torch.stack([image.flatten() for image, label in train_data]).to(device)
y_train = torch.tensor([label for image, label in train_data], dtype=torch.long, device=device)
X_test = torch.stack([image.flatten() for image, label in test_data]).to(device)
y_test = torch.tensor([label for image, label in test_data], dtype=torch.long, device=device)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
# KNN classification
k = 4
batch_size = 64
correct = 0
total = 0
start = time.time()
for start_idx in range(0, len(X_test), batch_size):
    end_idx = min(start_idx + batch_size, len(X_test))
    batch = X_test[start_idx:end_idx]
    # Calculate Euclidean distances
    distances = torch.cdist(batch, X_train)
    # Find 4 nearest neighbors
    nearest = torch.topk(
        distances,
        k=k,
        dim=1,
        largest=False
    ).indices
    # Get neighbor labels
    neighbor_labels = y_train[nearest]
    # Count votes for each class
    votes = torch.nn.functional.one_hot(
      neighbor_labels,
      num_classes=10
    ).sum(dim=1)
    predictions = votes.argmax(dim=1)
    correct += (predictions == y_test[start_idx:end_idx]).sum().item()
    total += len(batch)

classification_time = time.time() - start
accuracy = correct / total

print("\nKNN RESULTS")
print("K:", k)
print("Correct predictions:", correct)
print("Total predictions:", total)
print("Testing accuracy:", accuracy * 100, "%")
print("Classification time:", classification_time, "seconds")
