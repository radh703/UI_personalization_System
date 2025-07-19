import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import word_tokenize


import os

# Set NLTK data path FIRST
os.environ['NLTK_DATA'] = 'C:/Users/DELL/nltk_data'
nltk.data.path.append('C:/Users/DELL/nltk_data')

# Then download
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)

# Load data
df = pd.read_csv("10_first_line.csv")

def generate_output(row):
    likes, dislikes, improvements = [], [], []
    
    # --- Likes ---
    if "Minimalist" in str(row["Customization Choices"]):
        likes.append("minimalist themes")
    if row["Learning Style"]:
        likes.append(f"learning via {row['Learning Style']}")
    
    # --- Dislikes ---
    if row["Feedback"] == "Bad":
        dislikes.append("the current UX")
    if row["Internet Speed"] == "Slow":
        dislikes.append("slow internet speeds")
    
    # --- Improvements ---
    if row["Internet Speed"] == "Slow":
        improvements.append("Optimize page load speed for slow connections.")
    if "Minimalist" in str(row["Customization Choices"]):
        improvements.append("Offer more minimalist theme customization options.")
    if row["Learning Style"]:
        improvements.append(f"Recommend {row['Learning Style']}-based content prominently.")
    
    # Combine into output
    summary = []
    if likes:
        summary.append(f"This user prefers {', '.join(likes)}.")
    if dislikes:
        summary.append(f"They dislike {', '.join(dislikes)}.")
    if improvements:
        summary.append("To improve their experience: " + " ".join(improvements))
    
    return " ".join(summary)

# Apply to all rows
df["output_text"] = df.apply(generate_output, axis=1)

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

# Custom dataset class
class UXDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_text = (
            f"generate UX recommendation: "
            f"User ID: {row['User ID']} "
            f"Learning Style: {row['Learning Style']} "
            f"Customization Choices: {row['Customization Choices']} "
            f"Internet Speed: {row['Internet Speed']}"
        )
        
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        outputs = self.tokenizer(
            row["output_text"],
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = outputs["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels
        }

# Create training dataset
train_dataset = UXDataset(train_df, tokenizer)

# Load model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Function to generate recommendation (without prefix for evaluation)
def generate_recommendation(model, tokenizer, user_data):
    input_text = (
        f"generate UX recommendation: "
        f"User ID: {user_data['User ID']} "
        f"Learning Style: {user_data['Learning Style']} "
        f"Customization Choices: {user_data['Customization Choices']} "
        f"Internet Speed: {user_data['Internet Speed']}"
    )
    
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=256,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=2.5
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluation
print("Evaluating the model...")
references_bleu = []
hypotheses_bleu = []
references_rouge = []
hypotheses_rouge = []

for _, row in test_df.iterrows():
    generated = generate_recommendation(model, tokenizer, row)
    reference = row["output_text"]
    # For BLEU
    references_bleu.append([nltk.word_tokenize(reference)])
    hypotheses_bleu.append(nltk.word_tokenize(generated))
    # For ROUGE
    references_rouge.append(reference)
    hypotheses_rouge.append(generated)

# Calculate BLEU scores
corpus_bleu_score = corpus_bleu(references_bleu, hypotheses_bleu)
print(f"Corpus BLEU score: {corpus_bleu_score:.4f}")

bleu_scores = []
for ref, hyp in zip(references_bleu, hypotheses_bleu):
    score = sentence_bleu(ref[0], hyp, weights=(0.25, 0.25, 0.25, 0.25))
    bleu_scores.append(score)
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average sentence BLEU: {avg_bleu:.4f}")

# Calculate ROUGE scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

for ref, hyp in zip(references_rouge, hypotheses_rouge):
    scores = scorer.score(ref, hyp)
    for key in rouge_scores:
        rouge_scores[key].append(scores[key].fmeasure)

for key in rouge_scores:
    avg_score = sum(rouge_scores[key]) / len(rouge_scores[key])
    print(f"Average {key}: {avg_score:.4f}")

# Print sample outputs for manual inspection
print("\nSample generations:")
for i in range(min(3, len(test_df))):
    row = test_df.iloc[i]
    generated = generate_recommendation(model, tokenizer, row)
    reference = row["output_text"]
    print(f"Reference: {reference}")
    print(f"Generated: {generated}")
    print("-" * 50)