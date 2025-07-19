import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import os

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

# NLTK setup
os.environ['NLTK_DATA'] = 'C:/Users/DELL/nltk_data'
nltk.data.path.append('C:/Users/DELL/nltk_data')
nltk.download('punkt', quiet=True)

def generate_output(row):
    likes, dislikes, improvements = [], [], []

    if "Minimalist" in str(row["Customization Choices"]):
        likes.append("minimalist themes")
    if row["Learning Style"]:
        likes.append(f"learning via {row['Learning Style']}")

    if row["Feedback"] == "Bad":
        dislikes.append("the current UX")
    if row["Internet Speed"] == "Slow":
        dislikes.append("slow internet speeds")

    if row["Internet Speed"] == "Slow":
        improvements.append("Optimize page load speed for slow connections.")
    if "Minimalist" in str(row["Customization Choices"]):
        improvements.append("Offer more minimalist theme customization options.")
    if row["Learning Style"]:
        improvements.append(f"Recommend {row['Learning Style']}-based content prominently.")

    summary = []
    if likes:
        summary.append(f"This user prefers {', '.join(likes)}.")
    if dislikes:
        summary.append(f"They dislike {', '.join(dislikes)}.")
    if improvements:
        summary.append("To improve their experience: " + " ".join(improvements))

    return " ".join(summary)

# Load and prepare data
df = pd.read_csv("10_first_line.csv")
df["output_text"] = df.apply(generate_output, axis=1)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer setup
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens': ['[User]', '[Rec]']})

class UXGPT2Dataset(Dataset):
    def __init__(self, df, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []
        
        for _, row in df.iterrows():
            context = (
                f"[User] ID: {row['User ID']} | "
                f"Learning: {row['Learning Style']} | "
                f"Customization: {row['Customization Choices']} | "
                f"Internet: {row['Internet Speed']}"
            )
            text = f"{context}\n[Rec] {row['output_text']}{tokenizer.eos_token}"
            tokenized = tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            'input_ids': self.examples[idx]['input_ids'].squeeze(),
            'attention_mask': self.examples[idx]['attention_mask'].squeeze(),
            'labels': self.examples[idx]['input_ids'].squeeze()
        }

# Model setup
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Training arguments (removed evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./gpt2_results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    gradient_accumulation_steps=2,
    warmup_steps=50,
    logging_dir="./gpt2_logs",
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    learning_rate=3e-5,
    prediction_loss_only=True
)

class CustomCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        labels = batch["labels"]
        for i in range(len(labels)):
            try:
                rec_start = (labels[i] == tokenizer.convert_tokens_to_ids("[Rec]")).nonzero()[0]
                labels[i,:rec_start+1] = -100
            except IndexError:
                pass
        return batch

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=UXGPT2Dataset(train_df, tokenizer),
    data_collator=CustomCollator(tokenizer=tokenizer, mlm=False),
)

trainer.train()

def generate_gpt2_recommendation(user_data, tokenizer, model):
    context = (
        f"[User] ID: {user_data['User ID']} | "
        f"Learning: {user_data['Learning Style']} | "
        f"Customization: {user_data['Customization Choices']} | "
        f"Internet: {user_data['Internet Speed']}"
    )
    input_text = f"{context}\n[Rec]"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=256,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        num_beams=3,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    recommendation = full_output.split("[Rec]")[-1].strip()
    return f"{user_data['User ID']} — UX Recommendations: {recommendation}"

# Evaluation with original output format
if not df.empty:
    user_1 = df.iloc[0]
    print("\nSample Output:")
    print(generate_gpt2_recommendation(user_1, tokenizer, model))
    
    # Evaluation metrics
    print("\nEvaluation Results:")
    smooth = SmoothingFunction()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []
    
    for _, row in test_df.iterrows():
        generated = generate_gpt2_recommendation(row, tokenizer, model)
        ref_text = row["output_text"]
        gen_text = generated.split("UX Recommendations: ")[-1]
        
        ref_tokens = nltk.word_tokenize(ref_text.lower())
        gen_tokens = nltk.word_tokenize(gen_text.lower())
        
        bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smooth.method1)
        rouge = scorer.score(ref_text, gen_text)
        
        bleu_scores.append(bleu)
        rouge1_scores.append(rouge['rouge1'].fmeasure)
        rougeL_scores.append(rouge['rougeL'].fmeasure)
    
    print(f"Average BLEU: {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"Average ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
    print(f"Average ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}")

else:
    print("Error: No data in DataFrame")