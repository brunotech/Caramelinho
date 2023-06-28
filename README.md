## Caramelinho

<a target="_blank" href="https://colab.research.google.com/github/brunotech/Caramelinho/blob/main/Camarelinho.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</a><a target="_blank" href="https://github.com/brunotech/Caramelinho">
  <img src="https://img.shields.io/badge/-Github-blue?style=social&logo=github&link=https://github.com/brunotech/Caramelinho" alt="Github Project Page"/>
</a>

</br>


<!-- header start -->
<div style="width: 100%;">
    <img src="https://blog.cobasi.com.br/wp-content/uploads/2022/08/AdobeStock_461738919.webp" alt="Caramelinho" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<!-- header end -->

# Caramelinho

## Adapter Description
This adapter was created with the [PEFT](https://github.com/huggingface/peft) library and allowed the base model **Falcon-7b** to be fine-tuned on the [Canarim](https://huggingface.co/datasets/dominguesm/Canarim-Instruct-PTBR-Dataset) by using the method **QLoRA**.

## Model description
[Falcon 7B](https://huggingface.co/tiiuae/falcon-7b)

## 📣 Special Announcement: Introducing Caramelinho - A New Language Model in Portuguese 🍬

Have you heard about Caramelinho? We are excited to present to you the latest advancement in natural language processing in Portuguese. Developed with the help of the PEFT library and enhanced through the QLoRA method, Caramelo is the new language model that will tantalize your textual taste buds.

## 💡 Cutting-edge Technology:
Caramelinho was created based on the acclaimed Falcon-7b base model and fine-tuned using the PEFT library. This powerful combination has enabled advanced training of the model, ensuring even more precise understanding and refined contexts.

## 🌍 Comprehensive Dataset:
By utilizing the Canarim Instruct PTBR Dataset available at https://huggingface.co/datasets/dominguesm/Canarim-Instruct-PTBR-Dataset, Caramelinho has been trained on over 300,000 instructions in Portuguese. This extensive dataset provides a wealth of language knowledge, enabling Caramelo to excel in understanding and generating instructional content.

## ✨ QLoRA Method:
Through the application of the innovative QLoRA (Query Language Representation Adaptation) method, Caramelo excels in its ability to answer your questions with greater precision and offer intelligent solutions. The answers provided by Caramelo are formulated based on extensive and up-to-date knowledge, making it an indispensable tool for researchers, writers, and enthusiasts of the Portuguese language.

## ⚡ Experience Caramelinho:
We are thrilled to make Caramelinho available to you. Try out this new language model in Brazilian Portuguese and unlock the full potential of written communication. Whether you are working on AI projects, developing virtual assistants, or simply aiming to enhance your language skills, Caramelo is ready to be your trusted partner.



### Training results
| epoch | learning_rate | loss   | step |
|-------|---------------|--------|------|
| 0.19  | 0.0002        | 0.496  | 10   |
| 0.37  | 0.0002        | 0.4045 | 20   |
| 0.56  | 0.0002        | 0.3185 | 30   |
| 0.74  | 0.0002        | 0.3501 | 40   |
| 0.93  | 0.0002        | 0.2881 | 50   |
| 1.12  | 0.0002        | 0.2783 | 60   |
| 1.3   | 0.0002        | 0.2701 | 70   |
| 1.49  | 0.0002        | 0.208  | 80   |
| 1.67  | 0.0002        | 0.2175 | 90   |
| 1.86  | 0.0002        | 0.2248 | 100  |
| 2.05  | 0.0002        | 0.1415 | 110  |
| 2.23  | 0.0002        | 0.1788 | 120  |
| 2.42  | 0.0002        | 0.1748 | 130  |
| 2.6   | 0.0002        | 0.1839 | 140  |
| 2.79  | 0.0002        | 0.1778 | 150  |
| 2.98  | 0.0002        | 0.1986 | 160  |
| 3.16  | 0.0002        | 0.0977 | 170  |
| 3.35  | 0.0002        | 0.1209 | 180  |
| 3.53  | 0.0002        | 0.1328 | 190  |
| 3.72  | 0.0002        | 0.1503 | 200  |
| 3.91  | 0.0002        | 0.1649 | 210  |
| 4.09  | 0.0002        | 0.1284 | 220  |
| 4.28  | 0.0002        | 0.1156 | 230  |
| 4.47  | 0.0002        | 0.0689 | 240  |
| 4.65  | 0.0002        | 0.0885 | 250  |
| 4.84  | 0.0002        | 0.1168 | 260  |
| 5.02  | 0.0002        | 0.1102 | 270  |
| 5.21  | 0.0002        | 0.0619 | 280  |
| 5.4   | 0.0002        | 0.0767 | 290  |
| 5.58  | 0.0002        | 0.0922 | 300  |
| 5.77  | 0.0002        | 0.0591 | 310  |
| 5.95  | 0.0002        | 0.0893 | 320  |
| 6.14  | 0.0002        | 0.0562 | 330  |
| 6.33  | 0.0002        | 0.0541 | 340  |
| 6.51  | 0.0002        | 0.0629 | 350  |
| 6.7   | 0.0002        | 0.0612 | 360  |
| 6.88  | 0.0002        | 0.0526 | 370  |
| 7.07  | 0.0002        | 0.044  | 380  |
| 7.26  | 0.0002        | 0.0424 | 390  |
| 7.44  | 0.0002        | 0.0459 | 400  |
| 7.63  | 0.0002        | 0.0442 | 410  |
| 7.81  | 0.0002        | 0.039  | 420  |
| 8.0   | 0.0002        | 0.0375 | 430  |
| 8.19  | 0.0002        | 0.0315 | 440  |
| 8.37  | 0.0002        | 0.0348 | 450  |
| 8.56  | 0.0002        | 0.0324 | 460  |
| 8.74  | 0.0002        | 0.0382 | 470  |
| 8.93  | 0.0002        | 0.0257 | 480  |
| 9.12  | 0.0002        | 0.0361 | 490  |



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


