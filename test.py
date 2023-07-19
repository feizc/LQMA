from transformers import AutoModelForCausalLM 
import torch 
import os 


def test_gpt2_embeddding():
    model_path = 'ckpt/gpt2'
    gpt2 = AutoModelForCausalLM.from_pretrained(model_path) 
    print(gpt2.transformer.wte) 

def test_image_path(): 
    data_path = 'data/CUB' 
    g = os.walk(data_path) 
    for path, dir_list, file_list in g: 
        for file_name in file_list: 
            print(os.path.join(path, file_name))

# test_gpt2_embeddding() 
test_image_path() 
