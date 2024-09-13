from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)






'''
model_id = "meta-llama/Meta-Llama-3.1-8B"
#model_id = "facebook/opt-125m"
new_model = "finetunedlama_4bits"
access_token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_id,token=access_token)


quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",token=access_token)

prompt = "Who wrote the book innovator's dilemma?"
pipe = pipeline(task="text-generation", model=quantized_model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result)
print(result[0]['generated_text'])

quantized_model.to("cpu")
quantized_model.save_pretrained(f"finetuned/llama3.1/{new_model}")
'''