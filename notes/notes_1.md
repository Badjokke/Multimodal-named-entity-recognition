 # Session 1
 Notes after first meeting that took place on 15.07.2024.
> # Named Entity Recognition (NER)
> Subset of **information retrieval** which seeks to locate and classify named entities in unstructured text into pre-defined categories (persons, organizations, locations, ...)
>
>> example input: *Jim bought 300 shares of Acme Corp. in 2006.*
>>
>> example output: **Jim** \[Person] bought 300 shares of **Acme Corp.**\[Organization] in **2006**[Time].
> ## Definitions
>> ### Named Entity
>> real-world object which can be denoted with a *name*.
> ## Problems and solutions
>> + **Detection of names**
>>    + Tokenize the text, construct an ontology with the tokens -> get named entities? A lot of manual work \+ working with RDF (for example).
>>    + *How to label data?*
>> + **Classification of names**
>>    + Hopefully nothing new under the sun.
>>    + Machine learning -> requires labeled data set - one unknown how to do it, further research required.
>>    + Do we have enough data for complex neural networks (Reccurent, Transformes)? If yes then perfect else decision trees or SVM?
>>  
>> ## Viable techniques
>> + **Rule based system**
>>    + Out of scope, way too expensive and no one here is linguist.
>> + **Machine learning**
>>    + Very likely the way to go
>>    + **LSTMS**, **CRFs**, **Transformers**       
> # Relevant articles for further reading
>> + [Alignment Precedes Fusion: Open-Vocabulary Named Entity Recognition
as Context-Type Semantic Matching](https://aclanthology.org/2023.findings-emnlp.974.pdf)
> # Stuff read
>> + [IBM NER](https://www.ibm.com/topics/named-entity-recognition)
>> + [Wiki](https://en.wikipedia.org/wiki/Named-entity_recognition#Approaches)
> # Multi-modal NER
> Adding another modality (**image**, audio, ...)
> *Transfer learning* should kick in because natural language has shared properties across modalities.
> # **TODO READ THIS**
>> + [2M-NER: Contrastive Learning for Multilingual and
Multimodal NER with Language and Modal Fusion](https://arxiv.org/pdf/2404.17122)
>> + [LLMs as Bridges: Reformulating Grounded Multimodal Named Entity
Recognition](https://arxiv.org/pdf/2402.09989v4)
>> + [Multimodal Named Entity Recognition for Short Social Media Posts
](https://aclanthology.org/N18-1078.pdf)
> # Dataset



> # Questions
> *Adding image of the text as additional input -> will it increase the accururacy of classification?*
>> Hypothesis 0: Adding another modality, such as an image of the text should increase the classification.
>> From the articles in the notes above, multi-modal NER **significantly outperforms** the classical NER models. 
