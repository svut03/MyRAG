from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class Generator:
    def __init__(self, tokenizer_r_name: str, reader_name: str):
        self.tokenizer_r = AutoTokenizer.from_pretrained(tokenizer_r_name)
        self.reader = AutoModelForCausalLM.from_pretrained(reader_name)
        self.pipeline = pipeline(
            model=self.reader,
            tokenizer=self.tokenizer_r,
            task="text-generation",
        )

    def generate(self, prompt: str, max_new_tokens=128, temperature=0.2):

        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
              temperature=temperature,
                do_sample=True
        )

        answer = self.pipeline(
            prompt,
            generation_config=self.generation_config,
            return_full_text=False,
            clean_up_tokenization_spaces=False,
        )

        return answer[0]["generated_text"]
