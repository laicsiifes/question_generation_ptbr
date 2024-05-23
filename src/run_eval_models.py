import warnings
import torch
import os
import pandas as pd
import json

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.models_utils import encode_test_batch, clean_predictions
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.evaluation_measures import compute_bleu, compute_bertscore, compute_meteor, compute_rouge


warnings.filterwarnings('ignore')


def collate(batch_):
    return {
        'id': [x['id_qa'] for x in batch_],
        'original_question': [x['question'] for x in batch_],
        'original_answer': [x['answer'] for x in batch_],
        'labels_with_answer': [f"question: {x['question']} answer: {x['answer']}" for x in batch_],
        'input_ids': torch.tensor([x['input_ids'] for x in batch_]),
        'attention_mask': torch.tensor([x['attention_mask'] for x in batch_])
    }

"""
    TO DO         
        Implementar a separação da pergunta e resposta e avaliar individualmente cada.
        Ver casos dos exemplos com respostas None no corpus
"""

if __name__ == '__main__':

    dataset_name = 'pira'
    # dataset_name = 'squad_pt_v2'

    list_models = [
        'ptt5_small',
        'ptt5_base',
        'ptt5_large',
        'flan_t5_small',
        'flan_t5_base',
        'flan_t5_large'
    ]

    num_epochs = 20

    input_max_len = 512
    output_max_len = 128

    batch_size = 16

    use_answer_input = True
    output_with_answer = False

    if use_answer_input is True and output_with_answer is True:
        print(f'\nInvalid Configuration: use_answer_input: {use_answer_input} and {output_with_answer}')
        exit(-1)

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

    print(f'\nNum Epochs: {num_epochs} -- Use Input Answer: {use_answer_input} '
          f'-- Output with answer: {output_with_answer}')

    test_dataset = dataset['test']

    print(f'\nDataset: {dataset_name}')

    print(f'\n  Test: {len(test_dataset)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device}')

    input_config = 'in_ctx_ans' if use_answer_input else 'in_ctx'
    output_config = 'out_question_answer' if output_with_answer else 'out_question'

    results_dir = f'../data/results/{dataset_name}/{input_config}_{output_config}/{num_epochs}'

    os.makedirs(results_dir, exist_ok=True)

    df_results = pd.DataFrame(
        columns=['Model', 'Rouge-1 F1', 'Rouge-2 F1', 'RougeL F1', 'BertScore F1', 'BLEU', 'METEOR'])

    for model_name in list_models:

        models_dir = f'../data/models/{dataset_name}/{model_name}/{input_config}_{output_config}/{num_epochs}'

        best_model_dir = models_training_dir = os.path.join(models_dir, 'best_model')

        print(f'\nModel: {model_name} -- Model Path: {best_model_dir}')

        if not os.path.exists(best_model_dir):
            continue

        tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(best_model_dir)

        model.to(device)

        print('\nTesting Model\n')

        tokenized_test_dataset = test_dataset.map(
            encode_test_batch,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={
                'tokenizer': tokenizer,
                'input_max_len': input_max_len,
                'use_answer_input': use_answer_input
            }
        )

        batched_ds = DataLoader(tokenized_test_dataset, batch_size=batch_size, collate_fn=collate)

        all_examples = []

        for batch in tqdm(batched_ds):

            predict_logits = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                num_beams=5, min_length=10,
                max_length=output_max_len, num_return_sequences=1, no_repeat_ngram_size=3,
                remove_invalid_values=True
            )

            decoded_predictions = tokenizer.batch_decode(predict_logits, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)

            generated_questions, generated_answers = clean_predictions(decoded_predictions, output_with_answer)

            if output_with_answer:
                pass

            for id_example, reference_question, reference_answer, generated_question, generated_answer in (
                    zip(batch['id'], batch['original_question'], batch['original_answer'], generated_questions,
                        generated_answers)):
                generated_answer = generated_answer if len(generated_answer) > 0 else None
                all_examples.append(
                    {
                        'id': id_example,
                        'reference_question': reference_question,
                        'reference_answer': reference_answer,
                        'generated_question': generated_question,
                        'generated_answer': generated_answer
                    }
                )

        all_predictions = [e['reference_question'] for e in all_examples]
        all_references = [e['generated_question'] for e in all_examples]

        bleu_score = compute_bleu(all_references, all_predictions)

        meteor_score = compute_meteor(all_references, all_predictions)

        rouge_scores = compute_rouge(all_references, all_predictions)

        bert_scores = compute_bertscore(all_references, all_predictions)

        df_results.loc[len(df_results)] = [model_name, rouge_scores['rouge1'], rouge_scores['rouge2'],
                                           rouge_scores['rougeL'], bert_scores['f1_score'], bleu_score, meteor_score]

        results_file_path = os.path.join(results_dir, f'results.csv')

        df_results.to_csv(path_or_buf=results_file_path, index=False)

        json_file_path = os.path.join(results_dir, f'{model_name}_predictions.json')

        with open(file=json_file_path, mode='w', encoding='utf-8') as file:
            json.dump(all_examples, file, indent=4)
