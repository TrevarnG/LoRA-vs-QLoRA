from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    RobertaForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
import torch
from peft import (
    get_peft_config,
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
    TaskType,
)

lora_config = LoraConfig(
    task_type=TaskType.QUESTION_ANS,  
    inference_mode=False,                    # Set to False for training
    r=8,                                     # Rank for LoRA
    lora_alpha=16,                           # Alpha parameter for scaling
    lora_dropout=0.1,                        
    target_modules=["query", "value"],       # Target modules for applying LoRA
)

dataset = load_dataset("rajpurkar/squad_v2")
tokenizer = AutoTokenizer.from_pretrained("roberta-large")

base_model = RobertaForQuestionAnswering.from_pretrained(
    "roberta-large",
    load_in_4bit=True,                     # Enable 4-bit quantization
    device_map="auto",                     # Automatically map model to available devices
)

base_model = prepare_model_for_kbit_training(base_model)
RobertaModel = get_peft_model(base_model, lora_config)
RobertaModel.print_trainable_parameters()

def preprocess_function(examples):
    # Tokenize question and context with truncation and padding, return offset mapping for alignment
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,  # Dynamically truncate the input instead of fixing max length
        padding="longest",  # Dynamically pad the inputs to the longest sequence in the batch
        max_length=None,  # Allow max length to be set dynamically based on the input size
        return_offsets_mapping=True  # Return offsets to find token-level start/end positions
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(inputs["offset_mapping"]):
        # If there is an answer for this example
        if len(examples["answers"][i]["text"]) > 0:
            # Get the start position in the context and calculate the end position
            answer_start = examples["answers"][i]["answer_start"][0]
            answer_text = examples["answers"][i]["text"][0]
            answer_end = answer_start + len(answer_text)

            # Initialize token positions
            token_start = None
            token_end = None

            # Iterate over the offset mapping to find token start and end positions
            for idx, (start, end) in enumerate(offsets):
                if start <= answer_start < end:
                    token_start = idx
                if start < answer_end <= end:
                    token_end = idx
                    break

            # If we found valid token positions, store them, else set to unanswerable (0)
            if token_start is not None and token_end is not None:
                start_positions.append(token_start)
                end_positions.append(token_end)
            else:
                start_positions.append(0)  # For unanswerable questions or invalid spans
                end_positions.append(0)
        else:
            # If no answer is available (unanswerable question), set positions to 0
            start_positions.append(0)
            end_positions.append(0)

    # Remove the offset mapping from the inputs, as it's no longer needed
    inputs.pop("offset_mapping")

    # Add the start and end positions to the tokenized inputs
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

def compute_metrics(p):
    pred_starts = p.predictions[0].argmax(axis=1)
    pred_ends = p.predictions[1].argmax(axis=1)
    
    true_starts = p.label_ids[0]  # start_positions
    true_ends = p.label_ids[1]    # end_positions

    # Exact Match: Compare predicted start/end with true start/end
    exact_matches = (pred_starts == true_starts) & (pred_ends == true_ends)
    exact_match = np.mean(exact_matches)

    # F1 score calculation (word overlap between prediction and true answer)
    f1_scores = []
    for i in range(len(pred_starts)):
        # Predicted and true spans for each example
        pred_span = set(range(pred_starts[i], pred_ends[i] + 1))
        true_span = set(range(true_starts[i], true_ends[i] + 1))

        # Calculate precision, recall, and F1
        if len(pred_span) == 0 or len(true_span) == 0:
            f1 = 1.0 if pred_span == true_span else 0.0
        else:
            overlap = len(pred_span & true_span)
            precision = overlap / len(pred_span)
            recall = overlap / len(true_span)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    average_f1 = np.mean(f1_scores)

    return {
        "exact_match": exact_match,
        "f1": average_f1,
    }

data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    warmup_ratio=0.2,          
    lr_scheduler_type="linear",  
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=5000,
    fp16=True,  # Enable mixed precision training for efficiency
)

torch.cuda.empty_cache()

trainer = Trainer(
    model=RobertaModel,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
eval_results = trainer.evaluate()

print(f"Evaluation Results: {eval_results}")
