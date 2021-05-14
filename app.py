import requests, json
from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerFast


class TextSummerizeInput(BaseModel):
    text_input:str = Field(
        ...,
        title= 'Text input',
        description= 'Input text want to summerize texts.',
    )

class TextSummerizeOutput(BaseModel):
    output: str

def summarizer(input: TextSummerizeInput) -> TextSummerizeOutput:
    """ Summarize texts """
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
    inputs = tokenizer([tokenizer.bos_token + input.text_input + tokenizer.eos_token])['input_ids'][0]

    model_url = 'https://train-mxysk1opgrzauh8ifw55-gpt2-train-teachable-ainize.endpoint.dev.ainize.ai/predictions/bart-ko-small-finetune' 

    headers = {'Content-Type': 'application/json; charset=utf-8'}
    response = requests.post(url=model_url, headers=headers, json={"text" : inputs})

    if response.status_code == 200:
        result = tokenizer.decode(response.json()[0], skip_special_tokens=True)
        return TextSummerizeOutput(output=result)
    else:
        print(f'Failed {response.text}')
        return TextSummerizeOutput(output='Failed summerize')