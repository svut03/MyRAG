from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


class Generator():
    def __init__(self,
                 tokenizer_r_name: str,
                 reader_name: str):
        self.tokenizer_r = AutoTokenizer.from_pretrained(tokenizer_r_name)
        self.reader = AutoModelForCausalLM.from_pretrained(reader_name)
        self.pipeline = pipeline(
        model=self.reader,
        tokenizer=self.tokenizer_r,
        task='text-generation',
    )
    
    def generate(self, prompt):
    
        answer = self.pipeline(prompt,
                    max_new_tokens = 128,
                    temperature = 0.2,
                    do_sample = True,
                    return_full_text=False)

        return answer[0]['generated_text']