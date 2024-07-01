import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

data = [
    {"description": "morning meeting", "type": "work", "duration": 90, "priority": "high", "tag": "meeting"},
    {"description": "call my brother", "type": "personal", "duration": 10, "priority": "high", "tag": "call"},
    {"description": "develop new feature", "type": "work", "duration": 60, "priority": "medium", "tag": "development"},
    {"description": "review intern task's code", "type": "work", "duration": 30, "priority": "low", "tag": "review"},
    {"description": "create meeting for next week", "type": "work", "duration": 5, "priority": "null", "tag": "meeting"},
    {"description": "help Ahmet to move out", "type": "personal", "duration": 180, "priority": "high", "tag": "none"},
    {"description": "end of week meeting", "type": "work", "duration": 45, "priority": "medium", "tag": "meeting"}
]

# Filter only work tasks
work_tasks = [task for task in data if task["type"] == "work"]

# Extract descriptions for embedding
descriptions = [task["description"] for task in work_tasks]

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Compute embeddings
embeddings = np.array([nlp(description).vector for description in descriptions])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Sort tasks based on cosine similarity to the first task
# Here we assume the first task is the anchor for sorting
similarities = similarity_matrix[0]
sorted_indices = np.argsort(-similarities)

# Reorder tasks based on sorted indices
sorted_work_tasks = [work_tasks[i] for i in sorted_indices]

# Print the sorted work tasks
sorted_work_tasks_json = json.dumps(sorted_work_tasks, indent=2)
print(sorted_work_tasks_json)
