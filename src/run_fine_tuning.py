import warnings
import torch
import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from src.models_utils import preprocess_function, prepare_compute_eval_metrics
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback


warnings.filterwarnings('ignore')


if __name__ == '__main__':

    dataset_name = 'pira'
    # dataset_name = 'squad_pt_v2'

    # model_name = 'ptt5_small'
    model_name = 'ptt5_base'
    # model_name = 'ptt5_large'
    # model_name = 'flan_t5_small'
    # model_name = 'flan_t5_base'
    # model_name = 'flan_t5_large'

    use_answer_input = False
    output_with_answer = True

    num_epochs = 20

    batch_size = 16

    use_fp16 = False

    if use_answer_input is True and output_with_answer is True:
        print(f'\nInvalid Configuration: use_answer_input: {use_answer_input} and {output_with_answer}')
        exit(-1)

    input_max_len = 512

    output_max_len = 40

    if output_with_answer:
        output_max_len = 120

    dataset = None

    if dataset_name == 'pira':
        dataset = load_dataset('paulopirozelli/pira')
        dataset = dataset.rename_column(original_column_name='abstract_translated_pt',
                                        new_column_name='context')
        dataset = dataset.rename_column(original_column_name='question_pt_origin',
                                        new_column_name='question')
        dataset = dataset.rename_column(original_column_name='answer_pt_validate',
                                        new_column_name='answer')
    elif dataset_name == 'squad_pt_v2':
        dataset = load_dataset('tiagofvb/squad2-pt-br-no-impossible-questions')
    else:
        print('\nERROR. DATASET NAME OPTION INVALID!')
        exit(-1)

    dataset = dataset.filter(lambda example: example['answer'] is not None and len(example['answer'].strip()) >= 1)

    print(f'\nModel: {model_name} -- Num Epochs: {num_epochs} -- Use Input Answer: {use_answer_input} '
          f'-- Output with answer: {output_with_answer}')

    model_checkpoint = None

    if model_name == 'ptt5_small':
        model_checkpoint = 'unicamp-dl/ptt5-small-portuguese-vocab'
    elif model_name == 'ptt5_base':
        model_checkpoint = 'unicamp-dl/ptt5-base-portuguese-vocab'
    elif model_name == 'ptt5_large':
        model_checkpoint = 'unicamp-dl/ptt5-large-portuguese-vocab'
        use_fp16 = True
        batch_size = 8
    elif model_name == 'flan_t5_small':
        model_checkpoint = 'google/flan-t5-small'
    elif model_name == 'flan_t5_base':
        model_checkpoint = 'google/flan-t5-base'
    elif model_name == 'flan_t5_large':
        model_checkpoint = 'google/flan-t5-large'
        use_fp16 = True
        batch_size = 8
    else:
        print('\nERROR. MODEL OPTION INVALID!')
        exit(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device} -- Use FP16: {use_fp16}')

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    model.to(device)

    input_config = 'in_ctx_ans' if use_answer_input else 'in_ctx'
    output_config = 'out_question_answer' if output_with_answer else 'out_question'

    models_dir = f'../data/models/{dataset_name}/{model_name}/{input_config}_{output_config}/'

    os.makedirs(models_dir, exist_ok=True)

    train_dataset = dataset['train']
    validation_dataset = dataset['validation']

    print(f'\nDataset: {dataset_name}')

    print(f'\n  Train: {len(train_dataset)}')
    print(f'  Validation: {len(validation_dataset)}')

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={
            'tokenizer': tokenizer,
            'input_max_len': input_max_len,
            'output_max_len': output_max_len,
            'use_answer_input': use_answer_input,
            'output_with_answer': output_with_answer
        }
    )

    tokenized_valid = validation_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={
            'tokenizer': tokenizer,
            'input_max_len': input_max_len,
            'output_max_len': output_max_len,
            'use_answer_input': use_answer_input,
            'output_with_answer': output_with_answer
        }
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    logging_eval_steps = len(tokenized_train) // batch_size

    models_training_dir = os.path.join(models_dir, 'training')

    train_args = Seq2SeqTrainingArguments(
        output_dir=models_training_dir,
        num_train_epochs=num_epochs,
        learning_rate=5.6e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        eval_steps=logging_eval_steps,
        logging_steps=logging_eval_steps,
        evaluation_strategy='epoch',
        predict_with_generate=True,
        save_total_limit=1,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='rougeL',
        greater_is_better=True,
        push_to_hub=False,
        fp16=use_fp16
    )

    compute_eval_metrics = prepare_compute_eval_metrics(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_eval_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]

    )

    print('\nFine-tuning\n')

    trainer.train()

    best_model_dir = models_training_dir = os.path.join(models_dir, 'best_model')

    trainer.save_model(best_model_dir)

    tokenizer.save_pretrained(best_model_dir)

    results = trainer.evaluate()

    print(f'\nEval Results: {results}')
