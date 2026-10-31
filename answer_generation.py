import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from hybrid_retrieve_rerank import HybridRetrieveRerank

class QwenGeneratorRetail:
    def __init__(self, model_id="Qwen/Qwen2.5-3B-Instruct"):
        print(f"Loading Qwen Hugging Face Model: {model_id}")

        self.docRetriever = HybridRetrieveRerank()

        # 1. Automatically detect available hardware (Mac MPS, NVIDIA CUDA, or CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
            print("Using NVIDIA CUDA GPU.")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("Successfully moved model to Mac MPS (GPU) memory.")
        else:
            self.device = "cpu"
            print("Using CPU.")

        # Using huggingface model loading
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16 if self.device != "cpu" else torch.float32
        ).to(self.device)
            
        
        # Parameters using for response
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer = self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            device=None
        )

    def generate(self, query: str, top_k: int = 5) -> str:
        retrieved_chunks = self.docRetriever.search(query, top_res=top_k)
        # Citation Formatting and organize chunk
        context_citations = []
        for chunk in retrieved_chunks:
            # Check multiple common key names for document, page, and text
            doc = chunk.get("doc_name") or chunk.get("document_name") or "UnknownDoc"
            page = chunk.get("page_num") or chunk.get("evidence_page_num") or "X"
            text = chunk.get("text") or chunk.get("content") or ""
                    
            context_citations.append(f"Source: [{doc}, Page {page}]\nContent: {text}")


        context_str = "\n--------------------------".join(context_citations)


        # MAking the prompt
        # Qwen Chat Format
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise financial assistant. Answer the question using ONLY the provided document passages below. "
                    "If calculations are required, show your step-by-step math clearly. "
                    "Every fact or number must include a citation in the format [DocName, Page X]. "
                    "If the passages do not contain enough information to answer, state that you cannot answer from the available documents."
                ),
            },
            {
                "role": "user",
                "content": f"RETRIEVED PASSAGES:\n{context_str}\n\nQUESTION: {query}",
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


        output = self.pipe(prompt)
        response = output[0]["generated_text"]
        if "<|im_start|>assistant\n" in response:
            answer = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        else:
            answer = response.strip()
        return answer

if __name__ == "__main__":
    generator = QwenGeneratorRetail()
    test_query = "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."
    print(f"\n--- Testing Question: {test_query} ---\n")
    print(generator.generate(test_query))