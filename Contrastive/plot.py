import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Assuming you have your embeddings stored in REAL and FAKE folders
REAL_FOLDER = r"embeddings\k_4\Real"
FAKE_FOLDER = r"embeddings\k_4\Fake"

# Load embeddings from REAL folder
real_embeddings = []
for file in os.listdir(REAL_FOLDER):
    if file.endswith(".npy"):
        embedding = np.load(os.path.join(REAL_FOLDER, file))
        real_embeddings.append(embedding.reshape(-1))

# Load embeddings from FAKE folder
fake_embeddings = []
for file in os.listdir(FAKE_FOLDER):
    if file.endswith(".npy"):
        embedding = np.load(os.path.join(FAKE_FOLDER, file))
        fake_embeddings.append(embedding.reshape(-1))

# Convert lists to numpy arrays
real_embeddings = np.array(real_embeddings)
fake_embeddings = np.array(fake_embeddings)

# Concatenate real and fake embeddings
all_embeddings = np.concatenate((real_embeddings, fake_embeddings), axis=0)
from random import shuffle
# shuffle(all_embeddings)
# Create labels for real and fake embeddings
real_labels = np.zeros(real_embeddings.shape[0])
fake_labels = np.ones(fake_embeddings.shape[0])
labels = np.concatenate((real_labels, fake_labels))

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(all_embeddings)

# Plot t-SNE
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap=plt.cm.get_cmap("coolwarm", 2))
# plt.colorbar(ticks=range(2), label='Real vs Fake')
plt.title('t-SNE Plot of Embeddings BiFormer w/o Contrastive with k=4 on CiFAKE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
