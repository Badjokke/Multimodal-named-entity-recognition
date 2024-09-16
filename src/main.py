from model.builder.model_factory import(create_quant_model,create_quant_model_classifier)
from model.quantization_config import (create_default_quantization_config)
from model.train.trainer import (train, train_class)
from model.util.model_util import (get_max_length)
from data.dataset import (load_hf_dataset)
from data.dataset_util import preprocess_dataset, preprocess_dataset_class


def ner_classify():
    output_dir = "finetuned/llama_8b/final_checkpoint"
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    model,tokenizer = create_quant_model_classifier(model_name,create_default_quantization_config())
    ds = preprocess_dataset_class(load_hf_dataset(), tokenizer)
    

    train_class(model,tokenizer,ds)
    
    
if __name__ == "__main__":
    ner_classify()
    """
    output_dir = "finetuned/llama_8b/final_checkpoint"
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    ds = load_hf_dataset()
    model,tokenizer = create_quant_model(model_name,create_default_quantization_config(),"llaama_tuned")
    ds = preprocess_dataset(tokenizer,get_max_length(model),42,ds)
    
    train(dataset=ds,model=model,tokenizer=tokenizer,output_dir=output_dir)
    """
    '''
    model,tokenizer = create_quant_model(model_name,create_default_quantization_config(),"llaama_tuned")
    prompt = "Who wrote the book innovator's dilemma?"
    pipe = create_pipe("text-generation",model,tokenizer)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result)
    print(result[0]['generated_text']) 
    model.save_pretrained("finetuned/llama_8b")
    print("done, exiting")
    '''