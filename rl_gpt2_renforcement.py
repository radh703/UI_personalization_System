import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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

# Define the generate_output function
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

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens': ['[User]', '[Rec]']})
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.eval()

# Define the RL Agent
class RLAgent(nn.Module):
    def __init__(self):
        super(RLAgent, self).__init__()
        # Policy network to adjust generation parameters
        self.policy = nn.Sequential(
            nn.Linear(3, 64),  # Input: 3 features (learning_style, customization, internet_speed)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # Output: 4 possible action combinations
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.policy(state)

# Reward function based on BLEU and ROUGE scores
def compute_reward(generated_text, reference_text):
    smooth = SmoothingFunction()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    ref_tokens = nltk.word_tokenize(reference_text.lower())
    gen_tokens = nltk.word_tokenize(generated_text.lower())
    
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smooth.method1)
    rouge = scorer.score(reference_text, generated_text)
    
    # Combine BLEU and ROUGE-L scores for reward
    reward = 0.5 * bleu + 0.5 * rouge['rougeL'].fmeasure
    return reward

# Generate recommendation with adjustable parameters
def generate_gpt2_recommendation(user_data, tokenizer, model, temperature=0.7, top_k=50):
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
        temperature=temperature,
        top_k=int(top_k),
        top_p=0.9,
        num_beams=3,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    recommendation = full_output.split("[Rec]")[-1].strip()
    return recommendation

# Encode user data as state for RL
def encode_user_data(user_data):
    # Encode categorical data to numerical, excluding User ID
    learning_style = {'Visual': 0, 'Auditory': 1, 'Kinesthetic': 2}.get(user_data['Learning Style'], 0)
    internet_speed = {'Fast': 1, 'Slow': 0}.get(user_data['Internet Speed'], 0)
    customization = 1 if "Minimalist" in str(user_data['Customization Choices']) else 0
    return torch.tensor([learning_style, customization, internet_speed], dtype=torch.float32)

# RL Training Loop
def train_rl_agent(df, agent, optimizer, episodes=50):
    rewards_history = []
    # Predefined temperature and top_k combinations
    action_space = [
        (0.5, 30),  # Low temperature, low top_k
        (0.5, 70),  # Low temperature, high top_k
        (1.0, 30),  # High temperature, low top_k
        (1.0, 70),  # High temperature, high top_k
    ]
    
    for episode in range(episodes):
        episode_reward = 0
        log_probs = []
        rewards = []
        
        for _, row in df.iterrows():
            # Encode user data as state
            state = encode_user_data(row)
            
            # Get action probabilities
            action_probs = agent(state)
            dist = Categorical(action_probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            
            # Map action index to temperature and top_k
            temperature, top_k = action_space[action_idx.item()]
            
            # Generate recommendation
            generated_rec = generate_gpt2_recommendation(row, tokenizer, model, temperature, top_k)
            reward = compute_reward(generated_rec, row['output_text'])
            
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward
        
        # Compute discounted rewards
        discounted_rewards = []
        gamma = 0.99
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Update policy
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        policy_loss = torch.stack(policy_loss).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        rewards_history.append(episode_reward / len(df))
        print(f"Episode {episode + 1}/{episodes}, Average Reward: {rewards_history[-1]:.4f}")
    
    return rewards_history

# Load data
df = pd.read_csv("10_first_line.csv")
df["output_text"] = df.apply(generate_output, axis=1)

# Initialize RL agent and optimizer
agent = RLAgent()
optimizer = optim.Adam(agent.parameters(), lr=0.001)

# Train RL agent
rewards = train_rl_agent(df, agent, optimizer, episodes=50)

# Evaluate final performance
print("\nFinal Evaluation:")
bleu_scores = []
rouge1_scores = []
rougeL_scores = []
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
smooth = SmoothingFunction()

for _, row in df.iterrows():
    state = encode_user_data(row)
    action_probs = agent(state)
    action_idx = action_probs.argmax()
    temperature, top_k = [(0.5, 30), (0.5, 70), (1.0, 30), (1.0, 70)][action_idx.item()]
    generated_rec = generate_gpt2_recommendation(row, tokenizer, model, temperature, top_k)
    
    ref_text = row["output_text"]
    bleu = sentence_bleu([nltk.word_tokenize(ref_text.lower())], nltk.word_tokenize(generated_rec.lower()), smoothing_function=smooth.method1)
    rouge = scorer.score(ref_text, generated_rec)
    
    bleu_scores.append(bleu)
    rouge1_scores.append(rouge['rouge1'].fmeasure)
    rougeL_scores.append(rouge['rougeL'].fmeasure)

print(f"Final Average BLEU: {sum(bleu_scores)/len(bleu_scores):.4f}")
print(f"Final Average ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
print(f"Final Average ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}")