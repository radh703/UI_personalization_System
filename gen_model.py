import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import openai

# 1. CONFIGURATION
CSV_PATH = "/mnt/data/enhanced_ui_preference_dataset.csv"  # your CSV file with features only
OUTPUT_DIR = "./ui-recommender-output"
MODEL_NAME = "t5-small"  # or "t5-base"
TRAIN_RATIO = 0.8
EVAL_RATIO = 0.1
TEST_RATIO = 0.1
BATCH_SIZE = 8
NUM_EPOCHS = 3
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 150

# 2. SET OPENAI API KEY (for synthetic target generation)
openai.api_key = os.getenv("OPENAI_API_KEY")  # ensure this environment variable is set

# 3. LOAD DATA
df = pd.read_csv(CSV_PATH)

# 4. SYNTHETICALLY GENERATE TARGETS IF MISSING
def serialize_features(row):
    """Serialize feature row into descriptive prompt."""
    features = row.drop(labels=['User ID', 'Feedback'], errors='ignore')
    return ", ".join(f"{col}={row[col]}" for col in features.index)

if 'Recommendation' not in df.columns:
    print("Generating synthetic recommendations via OpenAI API...")
    def gen_rec(row):
        prompt = (
            "You are a UI/UX expert. Given the user context and preferences, "
            f"provide a concise paragraph with actionable UI personalization recommendations. "
            f"Context: {serialize_features(row)}.\nRecommendation:"  
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()

    df['Recommendation'] = df.apply(gen_rec, axis=1)
    df.to_csv("augmented_ui_preferences.csv", index=False)
    print("Saved augmented dataset with recommendations.")

# 5. SERIALIZE & PREPROCESS DATA
def serialize_row_for_t5(row):
    """Turn a dataframe row into T5 input prompt."""
    feat_str = serialize_features(row)
    return f"User: {feat_str}\nRecommendation:"  

def preprocess(df):
    inputs = df.apply(serialize_row_for_t5, axis=1).tolist()
    targets = df['Recommendation'].tolist()
    return inputs, targets

full_inputs, full_targets = preprocess(df)

# 6. SPLIT DATA
dataset_size = len(df)
train_size = int(TRAIN_RATIO * dataset_size)
eval_size = int(EVAL_RATIO * dataset_size)
test_size = dataset_size - train_size - eval_size

# 7. DATASET CLASS
class UIRecommendationDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer):
        self.enc = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        self.tgt = tokenizer(
            targets,
            max_length=MAX_TARGET_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.tgt.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.enc.input_ids[idx],
            'attention_mask': self.enc.attention_mask[idx],
            'labels': self.tgt.input_ids[idx],
        }

# 8. INSTANTIATE TOKENIZER & MODEL
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# 9. BUILD SPLIT DATASETS
all_dataset = UIRecommendationDataset(full_inputs, full_targets, tokenizer)
train_ds, eval_ds, test_ds = random_split(
    all_dataset, [train_size, eval_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 10. TRAINING SETUP
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
    predict_with_generate=True,
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)

# 11. TRAIN & EVALUATE
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Starting fine-tuning...")
    trainer.train()
    print("Evaluating on test set...")
    metrics = trainer.evaluate(test_ds)
    print(metrics)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")
