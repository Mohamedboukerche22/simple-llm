from transformes import pipeline

generator = pipeline(task="text-generation", model="gpt2")

review=text

response = "Dear valued customer, I am glad to hear you had a good stay with us."


prompt = f"Customer review:\n{review}\n\nHotel reponse to the customer:\n{response}"


outputs = generator(prompt, max_length=150, pad_token_id=generator.tokenizer.eos_token_id, truncation=True)

print(outputs[0]["generated_text"])
