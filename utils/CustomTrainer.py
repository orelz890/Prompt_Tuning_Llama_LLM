import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
from sentence_transformers import SentenceTransformer, util

# Define the custom trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, similarity_model=None, persona_classifier=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity_model = similarity_model
        self.persona_classifier = persona_classifier

    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs.keys())
        
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Token loss (CrossEntropyLoss)
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        token_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Diversity loss
        generated_ids = torch.argmax(logits, dim=-1).tolist()
        ngram_size = 3
        ngrams = [tuple(generated_ids[i:i+ngram_size]) for i in range(len(generated_ids) - ngram_size + 1)]
        unique_ngrams = set(ngrams)
        diversity_loss = 1 - (len(unique_ngrams) / max(1, len(ngrams)))

        # Persona consistency loss (if a classifier is provided)
        persona_loss = 0
        if self.persona_classifier:
            persona_logits = self.persona_classifier(logits)
            persona_targets = inputs.get("persona_labels")
            persona_loss = CrossEntropyLoss()(persona_logits, persona_targets)

        # Conversational relevance loss (using SentenceTransformer)
        relevance_loss = 0
        if self.similarity_model:
            input_text = inputs["input_text"]
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            input_embedding = self.similarity_model.encode(input_text, convert_to_tensor=True)
            response_embedding = self.similarity_model.encode(generated_text, convert_to_tensor=True)
            relevance_loss = 1 - util.cos_sim(input_embedding, response_embedding).mean()

        # Combine all losses
        alpha, beta, gamma, delta = 0.6, 0.2, 0.1, 0.1
        total_loss = (
            alpha * token_loss +
            beta * diversity_loss +
            gamma * persona_loss +
            delta * relevance_loss
        )

        return (total_loss, outputs) if return_outputs else total_loss
