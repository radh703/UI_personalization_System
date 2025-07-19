import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 1. Load and prepare data
df = pd.read_csv("10_first_line.csv")

def generate_output(row):
    likes, dislikes, improvements = [], [], []

    # Likes
    if "Minimalist" in str(row["Customization Choices"]):
        likes.append("minimalist themes")
    if row["Learning Style"]:
        likes.append(f"learning via {row['Learning Style']}")

    # Dislikes
    if row["Feedback"] == "Bad":
        dislikes.append("the current UX")
    if row["Internet Speed"] == "Slow":
        dislikes.append("slow internet speeds")

    # Improvements
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

df["output_text"] = df.apply(generate_output, axis=1)

# 2. Improved tokenizer setup
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens': ['[User]', '[Rec]']})

# 3. Enhanced dataset class with proper context formatting
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

dataset = UXGPT2Dataset(df, tokenizer)

# 4. Model setup with adjusted token embeddings
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# 5. Enhanced training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_results",
    num_train_epochs=7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    warmup_steps=50,
    logging_dir="./gpt2_logs",
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    learning_rate=3e-5,
    prediction_loss_only=True
)

# 6. Custom data collator for better masking
class CustomCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        # Mask context part for loss calculation
        labels = batch["labels"]
        for i in range(len(labels)):
            try:
                rec_start = (labels[i] == tokenizer.convert_tokens_to_ids("[Rec]")).nonzero()[0]
                labels[i,:rec_start+1] = -100  # Ignore context in loss
            except IndexError:
                pass
        return batch

data_collator = CustomCollator(tokenizer=tokenizer, mlm=False)

# 7. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

# 8. Improved generation function
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

# Test with first user
if not df.empty:
    user_1 = df.iloc[0]
    print(generate_gpt2_recommendation(user_1, tokenizer, model))
else:
    print("Error: No data in DataFrame")