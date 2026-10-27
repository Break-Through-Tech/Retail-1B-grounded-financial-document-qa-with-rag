import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from hybrid_retrieve_rerank import HybridRetrieveRerank

class LlamaGeneratorRetail:
    def __init__(self, model_id="meta-llama/Llama-3.2-3B-Instruct"):
        print(f"Loading Llama Hugging Face Model: {model_id}")

        self.docRetriever = HybridRetrieveRerank()

        # Using huggingface model loading
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype= torch.flot16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        # Parameters using for response
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer = self.tokenizer,
            max_tokens=512,
            temperature=0.1,
            do_sample=True
        )