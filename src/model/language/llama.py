import torch
class LlamaLM:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_prompt(self, text: str) -> str:
        """Prepare the prompt for NER task."""
        return f"""Extract named entities from the following text. Classify each entity as PERSON, OTHER, MISC or ORGANIZATION. Text: {text} Entities (format: Entity | Type):"""

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        """Extract named entities from the given text."""
        # Prepare input
        prompt = self.prepare_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                temperature=0.1,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token
            )

        # Decode and parse response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        # Parse entities
        entities = []
        for line in response.split("\n"):
            if "|" in line:
                entity, type_ = line.split("|")
                entities.append((entity.strip(), type_.strip()))

        return entities