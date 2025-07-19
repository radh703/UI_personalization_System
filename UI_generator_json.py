import json
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load Dataset
dataset_path = "enhanced_ui_preference_dataset.csv"
df = pd.read_csv(dataset_path)


print(df.head())

# Preprocess Data: Convert relevant columns to JSON-compatible format
def preprocess_data(row):
    """Convert a dataset row into a prompt-response format for model training."""
    input_text = f"Generate a UI configuration for {row['Layout Preference']} mode."
    output_json = {
        "theme": row["UI Theme"],
        "layout": json.loads(row["Layout Preference"]),  
        "colors": json.loads(row["Color Palette"])
    }
    return input_text, json.dumps(output_json)

# Tokenizer & Model Setup
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def tokenize_data(input_texts, output_texts):
    """Tokenizes input-output pairs for training."""
    inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    outputs = tokenizer(output_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return inputs, outputs

# Prepare Training Data
input_texts, output_texts = zip(*df.apply(preprocess_data, axis=1))
inputs, outputs = tokenize_data(input_texts, output_texts)

def train_model(model, inputs, outputs, epochs=3):
    """Fine-tune the model for JSON generation."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs_model = model(input_ids=inputs['input_ids'], labels=outputs['input_ids'])
        loss = outputs_model.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


train_model(model, inputs, outputs)


model.save_pretrained("/mnt/data/ui_json_generator")
tokenizer.save_pretrained("/mnt/data/ui_json_generator")

# Generate New JSON from Prompt
def generate_ui_json(prompt):
    """Generate a UI JSON configuration from a text prompt."""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=512)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example Usage
prompt = "Generate a UI configuration for dark mode."
print("Generated JSON:", generate_ui_json(prompt))
