from transformners import pipeline

spanish_text = "Este curso sobre LLMs se está poniendo muy interesante"


translator = pipeline(task="translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en")


translations = translator(spanish_text, clean_up_tokenization_spaces=True)

print(translations[0]["translation_text"])
