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

X_centered = X_train - mean
# Calculate covariance matrix
covariance = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

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
C = (
	torch.searchsorted(
			cumulative_variance,
			torch.tensor(0.95, device=device)
	).item() + 1
)

print("Number of PCA components:", C)
print("Explained variance:",cumulative_variance[C-1].item() * 100,"%")

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


# Use CPU tensors for tree operations
Xtr = X_train_pca.cpu()
Xte = X_test_pca.cpu()

ytr = y_train.cpu()
yte = y_test.cpu()

# Binary decision tree
class DecisionTree:
	def __init__(
		self,
		max_depth=10,
		min_samples=50,
		n_features=32,
		n_thresholds=9
	):
		self.max_depth = max_depth
		self.min_samples = min_samples
		self.n_features = n_features
		self.n_thresholds = n_thresholds
		self.nodes = []

	def gini(self, labels):
		counts = torch.bincount(
				labels, minlength=10
		).float()
		probabilities = counts / len(labels)
		return 1.0 - torch.sum(probabilities ** 2).item()

	def build_tree(self, X, y, depth=0):
			node_id = len(self.nodes)
			# Majority class at current node
			prediction = torch.bincount(
					y, minlength=10
			).argmax().item()
			self.nodes.append({
					"prediction": prediction,
					"feature": None,
					"threshold": None,
					"left": None,
					"right": None
			})
			# Stopping conditions
			if (
				depth >= self.max_depth
				or len(y) < 2 * self.min_samples
				or torch.unique(y).numel() == 1
			):
				return node_id

			parent_gini = self.gini(y)
			best_gain = 0
			best_feature = None
			best_threshold = None

			# Examine a subset of PCA features
			num_features = min(self.n_features, X.shape[1])
			feature_indices = torch.randperm(
					X.shape[1]
			)[:num_features]
			quantiles = torch.linspace(0.1, 0.9, self.n_thresholds)

			# Find the best split
			for feature in feature_indices:
					values = X[:, feature]
					thresholds = torch.quantile(values, quantiles).unique()
					for threshold in thresholds:
							left_mask = values <= threshold
							left_count = left_mask.sum().item()
							right_count = len(y) - left_count
							if (left_count < self.min_samples or right_count < self.min_samples):
								continue
							left_gini = self.gini(y[left_mask])
							right_gini = self.gini(y[~left_mask])
							weighted_gini = (left_count * left_gini + right_count * right_gini) / len(y)
							gain = parent_gini - weighted_gini
							if gain > best_gain:
									best_gain = gain
									best_feature = feature.item()
									best_threshold = threshold.item()

			# No useful split found
			if best_feature is None:
					return node_id

			# Divide data into two branches
			left_mask = X[:, best_feature] <= best_threshold

			self.nodes[node_id]["feature"] = best_feature
			self.nodes[node_id]["threshold"] = best_threshold

			# Build left subtree
			left_id = self.build_tree(
					X[left_mask],
					y[left_mask],
					depth + 1
			)
			# Build right subtree
			right_id = self.build_tree(
					X[~left_mask],
					y[~left_mask],
					depth + 1
			)
			self.nodes[node_id]["left"] = left_id
			self.nodes[node_id]["right"] = right_id
			return node_id
	def fit(self, X, y):
			self.nodes = []
			self.build_tree(X, y)

	def predict(self, X):
			predictions = torch.empty(len(X), dtype=torch.long)
			def traverse(node_id, sample_indices):
					node = self.nodes[node_id]
					# Leaf node
					if node["feature"] is None:
							predictions[sample_indices] = node["prediction"]
							return
					feature = node["feature"]
					threshold = node["threshold"]
					left_mask = (
							X[sample_indices, feature] <= threshold
					)
					left_indices = sample_indices[left_mask]
					right_indices = sample_indices[~left_mask]
					if len(left_indices) > 0:
						traverse(node["left"], left_indices)

					if len(right_indices) > 0:
						traverse(node["right"], right_indices)

			traverse(0,torch.arange(len(X)))
			return predictions

tree_start = time.time()
tree = DecisionTree(
    max_depth=10,
    min_samples=50,
    n_features=32,
    n_thresholds=9
)
tree.fit(Xtr, ytr)
tree_training_time = time.time() - tree_start
# Evaluate training accuracy
start = time.time()
train_predictions = tree.predict(Xtr)
tree_train_classification_time = time.time() - start
train_accuracy = (train_predictions == ytr).float().mean().item()
# Evaluate testing accuracy
start = time.time()
test_predictions = tree.predict(Xte)
tree_test_classification_time = time.time() - start
test_accuracy = (test_predictions == yte).float().mean().item()

# Number of nodes
node_count = len(tree.nodes)
print("Training accuracy:", train_accuracy * 100, "%")
print("Testing accuracy:", test_accuracy * 100, "%")
print("Number of tree nodes:", node_count)
print("Tree training time:",tree_training_time,"seconds")
print("Training classification time:",tree_train_classification_time,"seconds")
print("Testing classification time:",tree_test_classification_time,"seconds")

# Confusion matrix
cm = torch.zeros(10, 10, dtype=torch.int64)
indices = yte * 10 + test_predictions
cm += torch.bincount(
    indices,
    minlength=100
).reshape(10, 10)
print("\nDecision Tree Confusion Matrix:")
print(cm)
# Save results
torch.save({
    "tree": tree.nodes,
    "training_accuracy": train_accuracy,
    "testing_accuracy": test_accuracy,
    "number_of_nodes": node_count,
    "confusion_matrix": cm,
    "training_time": tree_training_time,
    "testing_time": tree_test_classification_time
}, "decision_tree_results.pth")
