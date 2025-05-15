from datasets import load_dataset
# For single-turn dataset
dataset = load_dataset("crag-mm-2025/crag-mm-single_turn-public", revision="v0.1.1")
# For multi-turn dataset
dataset = load_dataset("crag-mm-2025/crag-mm-multi_turn-public", revision="v0.1.1")
# View available splits
print(f"Available splits: {', '.join(dataset.keys())}")
# Access examples
example = dataset["validation"][0]
print(f"Session ID: {example['session_id']}")
print(f"Image: {example['image']}")
print(f"Image URL: {example['image_url']}")
"""
Note: Either 'image' or 'image_url' will be provided in the dataset, but not necessarily both.
When the actual image cannot be included, only the image_url will be available.
The evaluation servers will nevertheless always include the loaded 'image' field.
"""
# Show image
import matplotlib.pyplot as plt
plt.imshow(example['image'])
