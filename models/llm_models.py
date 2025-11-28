from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# Large model
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Base (Phi-2)
model2_name = "google/flan-t5-base"
tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
model2 = AutoModelForSeq2SeqLM.from_pretrained(model2_name)

# Small model
model3_name = "google/flan-t5-small"
tokenizer3 = AutoTokenizer.from_pretrained(model3_name)
model3 = AutoModelForSeq2SeqLM.from_pretrained(model3_name)

def run_model(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
