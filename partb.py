
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load KMNIST
transform = transforms.ToTensor()
train_data = torchvision.datasets.KMNIST("./data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.KMNIST("./data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

# Improved MLP
class ImprovedMLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.network = nn.Sequential(
			nn.Flatten(),
			nn.Linear(784, 256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, 10)
    )

def forward(self, x):
  return self.network(x)

model = ImprovedMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
losses = []
start = time.time()

for epoch in range(15):
	model.train()
	total_loss = 0

	for images, labels in train_loader:
		images, labels = images.to(device), labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	avg_loss = total_loss / len(train_loader)
	losses.append(avg_loss)
	print(f"Epoch {epoch+1}/15, Loss: {avg_loss:.4f}")

training_time = time.time() - start

# Evaluation
def evaluate(loader):
	model.eval()
	correct = 0
	total = 0
	cm = torch.zeros(10, 10, dtype=torch.int64)

	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			labels = labels.to(device)
			predictions = model(images).argmax(dim=1)
			correct += (predictions == labels).sum().item()
			total += labels.size(0)
			indices = (labels * 10 + predictions).cpu()
			cm += torch.bincount(indices, minlength=100
			).reshape(10, 10)
	return correct / total, cm

train_acc, train_cm = evaluate(train_loader)
test_acc, test_cm = evaluate(test_loader)

print("Training accuracy:", train_acc * 100, "%")
print("Testing accuracy:", test_acc * 100, "%")
print("Training time:", training_time, "seconds")

# Plot Confusion matrix
print("Confusion Matrix Values:")
print(test_cm)

# Plot loss
plt.plot(losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Improved MLP")
plt.show()