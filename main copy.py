from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import math
import textstat
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

max_sen_len=1024
no_of_sentences=5

user_input = "Cloud computing is a technology that allows users to access and store data, applications, and services over the internet instead of on local servers or personal devices. It offers scalable resources, enabling businesses and individuals to use computing power and storage as needed without investing in physical infrastructure. Cloud services are typically offered in three forms: Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). Benefits include cost savings, flexibility, and remote accessibility. Cloud computing has become integral to modern IT, driving innovation and efficiency across various industries."
sentences = sent_tokenize(user_input)

# model Load
model_name="unikei/t5-base-split-and-rephrase"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

print(f"Device:{device}")
def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
    return math.exp(loss.item())


def evaluate_readability(text):
  return {
      "Flesch-Kincaid": textstat.flesch_kincaid_grade(text),
      "SMOG": textstat.smog_index(text),
      "Gunning Fog": textstat.gunning_fog(text),
      "Flesch Reading Ease": textstat.flesch_reading_ease(text)
  }
output_text = {
        "text1": [],
        "text2": [],
        "text3": []
        }
for i, sentence in enumerate(sentences):
    # sentence = "Climate change refers to significant and lasting changes in the Earth's climate patterns, \
    # particularly those changes attributed to human activities."
    # print(f"sentence:{sentence}")
    input_ids = tokenizer(
        sentence,
        return_tensors="pt"
    ).input_ids

    # print("input id size:",input_ids.size())

    outputs = model.generate(
        input_ids,
        num_beams=10,
        num_return_sequences=no_of_sentences,
        max_length=max_sen_len, #only manx sen len can be outputed for a single sentence
        # min_length=20,            # Minimum length of generated sequence
        length_penalty=1.0,       # Encourages longer sentences if > 1.0
        do_sample=True,
        temperature=0.7,          # Controls randomness (lower is more deterministic)
        top_k=50,                 # Only considers top 50 tokens for sampling
        top_p=0.95,               # Only considers tokens with cumulative probability > 0.95
        early_stopping=True
    )

    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    results = []


    print("\n" + "="*50 + "\n")
    print(f"sentence {i+1}: {sentence}")
    print(f"\nOutput for sentence {i+1}:\n")
    for i, text in enumerate(generated_texts):

        perplexity = calculate_perplexity(text)
        readability_scores = evaluate_readability(text)
        results.append({
            'index': i + 1,
            'text': text,
            'perplexity': perplexity,
            'readability_scores': readability_scores
        })

    # sorted_results = sorted(results, key=lambda x: x['perplexity'],reverse=True)
    sorted_results = sorted(results, key=lambda x: x['readability_scores']['Flesch Reading Ease'],reverse=True)
    if sorted_results:
        print("***text has been appended.***")
        output_text["text1"].append(sorted_results[0]['text'])
        output_text["text2"].append(sorted_results[1]['text'])
        output_text["text3"].append(sorted_results[2]['text'])

    for result in sorted_results:
        print(f"\nDemo {result['index']}: {result['text']}")
        print(f"Perplexity: {result['perplexity']}")
        print("Readability Scores:")
        for metric, score in result['readability_scores'].items():
            print(f"{metric}: {score}")


    print("\n" + "="*50 + "\n")
    
    
for i, key in enumerate(output_text):
    merged_text = " ".join(output_text[key])
    print(f"Sample Output {i+1}:\n")
    print(merged_text)
    output_text[key] = merged_text

    print(f"Perplexity: {calculate_perplexity(merged_text)}")
    readability_scores = evaluate_readability(merged_text)
    print("Readability Scores:")
    for metric, score in result['readability_scores'].items():
        print(f"{metric}: {score}")
    print("\n" + "="*50 + "\n")