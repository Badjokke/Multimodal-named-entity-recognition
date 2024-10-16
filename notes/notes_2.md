# Session 2 
Previous sessions did not produce any huge output from my side therefore this session will summarize last month and a half.
 # Dataset exploration
 Provided dataset (Vlastivědy) contains several directories and files:   
- Heimatkunde-Plan_a15539
  - used by Honzik is his master´s thesis + unfit for multimodal purposes   
- Heimatskunde-Asch_r448
  - used by Honzik is his master´s thesis + unfit for multimodal purposes
- soap-so_knihovna_gnirs-topographie-denkmale-1927_pk0301-52
  - book is sadly not suitable for creating a dataset with exceptions:  
    - Chodau (Chodov) - small town with tower that can be used in dataset (page 21 in the book)
    - Elboge (Loket) - a lot of pictures of the castle Loket from different view points. Can be used in dataset
  - rest of the villages / towns are very generic and unfit.
- Other books in Vlastivědy (some attached pdfs) are not fit aswell. The largest one (The impact of Finnish culture on German folklore) has a lot of pages with nice images, that are sadly not related to the text itself (ie the characters or buildings are just represantional; they can´t be used to identify entity.)
 
# Explored Large Language Model
Llama 3.1 8B and Mistral 7B v0.1 were tried with quantization on (and with LoRa) - QLora.

Llama achieved accuracy of 75% on conll2003 dataset, Mistral 77%. Nothing to write home about.  
Further experiments required. 


# What to put in master´s thesis
**Quantization, Lora and Neural networks** (perceptron just as a building block, transformer in detail).  
Llama and Mistral for "used in experiments" - no need to write detailed information about them because there is not much to be said.

# What is to be done 
Find some nice multimodal dataset for NER. (high prio)   
Metacentrum. (medium prio)