import torch 
from torch.utils.data import DataLoader 
import numpy as np
#HuggingFace Stuff 
from transformers import AutoTokenizer, AutoModel
TODO: ADD IN ROBERTA INSTEAD OF GPT2
from transformers import GPT2TokenizerFast, GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2ForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset, load_metric  
# Misc  
from tqdm import tqdm 
import argparse 
import random
import os
import pickle 

#TODO: THIS IS THE FUNCTION THAT PUTS DATA INTO BATCHES. 
#TODO: 
def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch]) 
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    outputs = { "input_ids": input_ids, "attention_mask": input_mask}
    return outputs
      
def eval(args, data, model, tokenizer):
    # Set model to eval mode. Load metric and create data loader.  
    # If batch_size = 1 can get rid of collate_rn. 
    print("Evaluating") 
    eval_loader = DataLoader(data, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Send model to gpu if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    # Lists to store results. 
    preds_list = []
    probs_list = []
    
    # main EVAL loop 
    for idx, batch in enumerate(eval_loader):
        # send batch to GPU if available, else CPU
        batch = {key: value.to(device) for key, value in batch.items()} 
       
        # Set model to eval and run input batches with no_grad to disable gradient calculations   
        with torch.no_grad():
            outputs = model(**batch) 
            logits = outputs.logits   
       
        # Store Predictions
        preds = logits.argmax(axis=1)        
        preds = preds.detach().cpu().numpy()  
        preds_list.append(preds)   
        
        #Probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)  
        probs = probs.detach().cpu().numpy()  
        probs_list.append(probs)
        
    return probs_list, preds_list
    
    def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--optimizer", default=AdamW)
    parser.add_argument("--seed", default=83, type=str) 
    args = parser.parse_args() 
    
    # Load data 
    # TODO: CHANGE TO OUR DATA
    data = #TODO import data
    print("Data loaded") 
    
    #Sow seeds 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)

    #Instantiate the tokenizer and add the padding token 
    #TODO: CHANGE TO APPROPRIATE TOKENIZER
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") 
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token 
    
    #TODO: ADD IN THE ROBERTA MODEL FROM HUGGINGFACE:
    #TODO: LOAD MODEL WEIGHTS FOR PRE-TRAINED ROBERTA 
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=num_labels)
    
    #Specifying the pad token  
    model.config.pad_token_id = model.config.eos_token_id

    print("Preprocessing Data")
    #TODO: THIS FUNCTION WILL GET THE DATA IN APPROPRIATE FORMAT AS A DICTIONARY TOKENIZED TEXT
    #TODO: THIS NEEDS TO BE DONE FROM SCRATCH
    def preprocess(example): 
        tokenizer.pad_token = tokenizer.eos_token 
        results = tokenizer(*inputs, max_length=100, truncation=True, padding=True) 
        return results
 
    #CHANGE TO OUR DATASET
    eval_data = list(map(preprocess, data)) 
    results = eval(args=args, data=eval_data, model=model, tokenizer=tokenizer) 
    
if __name__  == "__main__":
    main()
    
    
    
