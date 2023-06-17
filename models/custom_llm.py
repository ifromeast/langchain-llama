import sys
from typing import Any, List, Dict, Mapping, Optional
import torch

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.requests import TextRequestsWrapper
from langchain.llms.base import LLM
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if torch.backends.mps.is_available():
    device = "mps"


def inference(model, tokenizer, batch_data, temperature=1, top_p=0.95, 
              top_k=40, num_beams=1, max_new_tokens=2048, **kwargs,):
    
    prompts = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {batch_data} ASSISTANT:"""
    
    inputs = tokenizer(prompts, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences       
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    output = output[0].split("ASSISTANT:")[1].strip()
    return output


class CustomLLM(LLM):
    logging: bool = True
    load_8bit: bool = False
    base_model: str = "decapoda-research/llama-13b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(base_model, truncation=True, )

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    @property
    def _llm_type(self) -> str:
        return self.model.config.model_type

    def log(self, log_str):
        if self.logging:
            print(log_str)
        else:
            return

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        self.log('----------' + self._llm_type + '----------> llm._call()')
        self.log(prompt)

        response = inference(self.model, self.tokenizer, prompt)
        return response
    

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": 10}


if __name__ == '__main__':
    llm = CustomLLM()
    print(llm("who are you?"))