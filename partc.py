
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
train_data = torchvision.datasets.KMNIST("./data", train=True,download=True,transform=transform)
test_data = torchvision.datasets.KMNIST("./data",train=False,download=True,transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=128,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=128,shuffle=False)

class CNN(nn.Module):
	def __init__(
			self,
			channels,
			batch_norm=True,
			dropout=0.0
	):
		super().__init__()
		layers = []
		in_channels = 1
		for out_channels in channels:
			layers.append(
				nn.Conv2d(
					in_channels,
					out_channels,
					kernel_size=3,
					padding=1
				)
			)
			if batch_norm:
				layers.append(
						nn.BatchNorm2d(out_channels)
				)
			layers.append(nn.ReLU())
			layers.append(nn.MaxPool2d(2))
			in_channels = out_channels
		self.features = nn.Sequential(*layers)
		with torch.no_grad():
			dummy = torch.zeros(1, 1, 28, 28)
			feature_size = self.features(dummy).numel()

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(dropout),
			nn.Linear(feature_size, 10)
		)

	def forward(self, x):
		x = self.features(x)
		return self.classifier(x)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    cm = torch.zeros(10, 10, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            indices = (labels * 10 + predictions).cpu()
            cm += torch.bincount(indices,minlength=100).reshape(10, 10)
    accuracy = correct / total
    return accuracy, cm


experiments = {
    "CNN 1 - Small": {
        "channels": [8, 16, 32],
        "batch_norm": True,
        "dropout": 0.0
    },
    "CNN 2 - Medium": {
        "channels": [16, 32, 64],
        "batch_norm": True,
        "dropout": 0.0
    },
    "CNN 3 - Large": {
        "channels": [32, 64, 128],
        "batch_norm": True,
        "dropout": 0.3
    },
    "CNN 4 - No BatchNorm": {
        "channels": [32, 64, 128],
        "batch_norm": False,
        "dropout": 0.3
    }
}

results = {}
for name, settings in experiments.items():
    print("Training:", name)
    model = CNN(
        channels=settings["channels"],
        batch_norm=settings["batch_norm"],
        dropout=settings["dropout"]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )
    start = time.time()
    losses = []
    # Train for 15 epochs
    for epoch in range(15):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = (total_loss / len(train_loader))
        losses.append(avg_loss)
        print(
            f"Epoch {epoch+1}/15, "
            f"Loss: {avg_loss:.4f}"
        )
    training_time = time.time() - start
    # Evaluate model
    train_acc, train_cm = evaluate(model, train_loader)
    test_acc, test_cm = evaluate(model, test_loader)
    print("\nRESULTS:", name)
    print("Training accuracy:", train_acc * 100, "%")
    print("Testing accuracy:", test_acc * 100, "%")
    print("Training time:", training_time, "seconds" )
    results[name] = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "training_time": training_time,
        "confusion_matrix": test_cm,
        "losses": losses,
        "model": model
    }
    # Save trained model
    torch.save(
        model.state_dict(),
        name.replace(" ", "_") + ".pth"
    )

print("\n\nCNN Comparison")
print(
    f"{'Architecture':<22}"
    f"{'Train Acc':<15}"
    f"{'Test Acc':<15}"
    f"{'Time (s)':<12}"
)

for name, result in results.items():
	print(
		f"{name:<22}"
		f"{result['train_accuracy']*100:<15.2f}"
		f"{result['test_accuracy']*100:<15.2f}"
		f"{result['training_time']:<12.2f}"
  )

best_name = max(results,key=lambda name: results[name]["test_accuracy"])
best_result = results[best_name]
print("\nBEST CNN:", best_name)
print(
	"Best testing accuracy:",
	best_result["test_accuracy"] * 100,
	"%"
)
print("\nBest CNN Confusion Matrix:")
print(best_result["confusion_matrix"])

# Save best model
torch.save(
	best_result["model"].state_dict(),
	"best_cnn.pth"
)

print("\nComparison")
print(
	f"{'Architecture':<22}"
	f"{'Train Acc (%)':<16}"
	f"{'Test Acc (%)':<16}"
	f"{'Time (s)':<12}"
)

for name, result in results.items():
	print(
		f"{name:<22}"
		f"{result['train_accuracy']*100:<16.2f}"
		f"{result['test_accuracy']*100:<16.2f}"
		f"{result['training_time']:<12.2f}"
	)

# Best CNN
best_name = max(resultskey=lambda name: results[name]["test_accuracy"])
best_result = results[best_name]
print("\nBest CNN:")
print("Best architecture:", best_name)
print("Training accuracy:",f"{best_result['train_accuracy']*100:.2f}%")
print("Testing accuracy:"f"{best_result['test_accuracy']*100:.2f}%")
print("Training time:"f"{best_result['training_time']:.2f} seconds")

# Best CM
cm = best_result["confusion_matrix"]
print("\nBest CNN Confusion Matrix:")
print("Rows = True Labels")
print("Columns = Predicted Labels")
print("    ", end="")
for j in range(10):
  print(f"{j:6}", end="")
print()
for i in range(10):
	print(f"{i:3} ", end="")
	for j in range(10):
		print(f"{cm[i, j].item():6}", end="")
	print()