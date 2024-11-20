# Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer from Hugging Face
model_name = "gpt2"  # Using the smaller GPT-2 model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad_token to eos_token since GPT-2 doesn't have a padding token by default
tokenizer.pad_token = tokenizer.eos_token

# Ensure the model is in evaluation mode
model.eval()

# Function to generate story from prompt
def generate_story(prompt, max_length=300):
    # Encode the input prompt to tokens and set the attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate a story based on the prompt
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],  # Pass attention mask
            max_length=max_length, 
            num_return_sequences=1, 
            do_sample=True,        # Enable sampling for creativity
            no_repeat_ngram_size=2, 
            temperature=0.7,      # Control randomness
            pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
        )

    # Decode the generated tokens to text
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

# Example prompt to start the story
prompt = "there lived a king long time ago"

# Generate the story
generated_story = generate_story(prompt)

# Print the generated story
print(generated_story)
