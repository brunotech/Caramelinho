datasets:
- dominguesm/Canarim-Instruct-PTBR-Dataset
library_name: adapter-transformers
pipeline_tag: text-generation
language:
- pt
- en

---

<a target="_blank" href="https://colab.research.google.com/github.com/brunotech/Caramelinho/blob/main/Camarelinho.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a><a target="_blank" href="https://github.com/brunotech/Caramelinho">
  <img src="https://img.shields.io/badge/-Github-blue?style=social&logo=github&link=https://github.com/brunotech/Caramelinho" alt="Github Project Page"/>
</a>

</br>

<!-- header start -->
<div style="width: 100%;">
    <img src="https://blog.cobasi.com.br/wp-content/uploads/2022/08/AdobeStock_461738919.webp" alt="Caramelo" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<!-- header end -->

# Caramelinho

## Adapter Description
This adapter was created with the [PEFT](https://github.com/huggingface/peft) library and allowed the base model **Falcon-7b** to be fine-tuned on the [Canarim](https://huggingface.co/datasets/dominguesm/Canarim-Instruct-PTBR-Dataset) by using the method **QLoRA**.

## Model description
[Falcon 7B](https://huggingface.co/tiiuae/falcon-7b)

## Intended uses & limitations
TBA

## Training and evaluation data
TBA

### Training results
TBA

### How to use
```python
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

peft_model_id = "Bruno/Caramelinho"
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                             return_dict=True,
                                             quantization_config=bnb_config, 
                                             trust_remote_code=True, 
                                             device_map={"": 0})

prompt_input = "Abaixo está uma declaração que descreve uma tarefa, juntamente com uma entrada que fornece mais contexto. Escreva uma resposta que conclua corretamente a solicitação.\n\n ### Instrução:\n{instruction}\n\n### Entrada:\n{input}\n\n### Resposta:\n"
prompt_no_input = "Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que conclua corretamente a solicitação.\n\n### Instrução:\n{instruction}\n\n### Resposta:\n"

def create_prompt(instruction, input=None):
    if input:
        return prompt_input.format(instruction=instruction, input=input)
    else:
        return prompt_no_input.format(instruction=instruction)

def generate(
        instruction,
        input=None,
        max_new_tokens=128,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        repetition_penalty=1.7,
        max_length=512
):
    prompt = create_prompt(instruction, input)
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt", truncation=True, max_length=max_length, padding="longest")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    generation_output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        length_penalty=0.8,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True
    )

    output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
    return output.split("### Resposta:")[1]

instruction = "Descrever como funcionam os computadores quânticos."
print("Instrução:", instruction)
print("Resposta:", generate(instruction))

### Saída
Instrução: Descrever como funcionam os computadores quânticos.
Resposta: 
Os computadores quânticos são um tipo de computador cuja arquitetura é baseada na mecânica quântica. Os computadores quânticos são capazes de realizar operações matemáticas complexas em um curto espaço de tempo.

### Framework versions
- Transformers 4.30.0.dev0
- Pytorch 2.0.1+cu118
- Datasets 2.12.0
- Tokenizers 0.13.3
