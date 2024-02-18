import transformers
from tokenizers import Tokenizer, Encoding
from transformers import LongT5Model
from typing import List
from datasets import load_dataset
import json
import hydra

@hydra.main(config_path='config', config_name='config',version_base=None)
def main(cfg):
    # Load the model
    model:transformers.LongT5Model = transformers.AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-local-base")

    # Load the processor
    tokenizer:transformers.T5Tokenizer = transformers.AutoTokenizer.from_pretrained("google/long-t5-local-base")
    with open("./gslm_tokenizer_hubert_tts/tokenizer.json") as f:
        vocabs = list(json.load(f)['model']['vocab'].keys())
        for vocab in vocabs:
            if vocab.isnumeric():
                tokenizer.add_tokens(f"speech_{vocab}")
            else:
                tokenizer.add_tokens(vocab)

    # Preprocess text
    train_dataset = load_dataset("json",data_files=cfg.ulm.ljspeech.dataset_path+"/train.json")
    val_dataset = load_dataset("json", data_files=cfg.ulm.ljspeech.dataset_path+"/dev.json")   
    test_dataset = load_dataset("json",data_files=cfg.ulm.ljspeech.dataset_path+"/test.json")

    def preprocess_function(examples):
        inputs = [f"{ex}" for ex in examples['phones']]
        targets = [" ".join([f"speech_{y}" for y in ex.split(" ")]) for ex in examples['feature']]
        model_inputs = tokenizer(inputs,text_target=targets,max_length=1024,truncation=True)
        return model_inputs
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model="google/long-t5-local-base")
    model.resize_token_embeddings(len(tokenizer))


    args = transformers.Seq2SeqTrainingArguments(
        output_dir="t5-"+cfg.ulm.ljspeech.dataset_path,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        logging_dir="./logs",   
        logging_steps=1000,
        warmup_steps=1000,
        max_steps=100_000,
        learning_rate=1e-4,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        metric_for_best_model="eval_loss",
        save_total_limit=1,
    )
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset['train'],
        eval_dataset=val_dataset['train']
    )
    trainer.train()

if __name__ == "__main__":
    main()