# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model_name = "google/pegasus-xsum"
# save_path = "./models/pegasus-xsum"   # local folder

# print("Downloading model...")

# # Load model + tokenizer from Hugging Face
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # Save them locally
# tokenizer.save_pretrained(save_path)
# model.save_pretrained(save_path)

# print(f"Model saved at {save_path}")



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/bart-large-cnn"
save_path = "./models2/bart-large-cnn"   # local folder

print("Downloading BART model...")

# Load from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Save locally
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Model saved at {save_path}")
