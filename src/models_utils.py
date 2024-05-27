import nltk
import evaluate
import numpy as np


def preprocess_function(examples, tokenizer, input_max_len, output_max_len,
                        use_answer_input=False, output_with_answer=False):
    """
        3 variações
            1. Apenas contexto e gerar pergunta use_answer = False, output_with_answer = False
            2. Contexto, resposta e gerar pergunta use_answer = True, output_with_answer = False
            3. Contexto e gerar pergunta e resposta use_answer = False, output_with_answer = True
    """

    list_answers = examples['answer']
    list_question = examples['question']

    list_contexts = examples['context']

    PREFIX = 'gere '
    if use_answer_input:
        input_contexts = [
            f'CONTEXT: {context}  ANSWER: {answer}'
            for context, answer in zip(list_contexts, list_answers)]
    else:
        input_contexts = [
            f'CONTEXT: {context}' for context in list_contexts
        ]

    if output_with_answer:
        output_questions = [
            f'QUESTION: {question}  ANSWER: {answer}'
            for question, answer in zip(list_question, list_answers)
        ]
    else:
        output_questions = [
            f'QUESTION: {question}' for question in list_question
        ]

    model_inputs = tokenizer(
        input_contexts,
        max_length=input_max_len,
        truncation=True,
        padding='max_length'
    )

    labels = tokenizer(
        text_target=output_questions,
        max_length=output_max_len,
        truncation=True,
        padding='max_length'
    )

    labels[labels == 0] = -100

    model_inputs['labels'] = labels['input_ids']

    return model_inputs


def encode_test_batch(examples, tokenizer, input_max_len: int, use_answer_input: bool = False):

    list_answers = examples['answer']

    list_contexts = examples['context']

    if use_answer_input:
        input_contexts = [f'CONTEXT: {context}</s>ANSWER: {answer}'
                          for context, answer in zip(list_contexts, list_answers)]
    else:
        input_contexts = [f'CONTEXT: {context}' for context in list_contexts]

    model_inputs = tokenizer(
        input_contexts,
        max_length=input_max_len,
        truncation=True,
        padding='max_length'
    )

    return model_inputs


nltk.download('punkt')
rouge = evaluate.load('rouge')


def prepare_compute_eval_metrics(tokenizer):
    def compute_eval_metrics(eval_pred) -> dict:
        nonlocal tokenizer
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = ['\n'.join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ['\n'.join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        result = rouge.compute(predictions=decoded_preds,
                               references=decoded_labels, use_stemmer=False)
        result = {key: value for key, value in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result['gen_len'] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}
    return compute_eval_metrics


def convert_predictions(list_predictions: list) -> tuple[list[str], list[str]]:
    list_questions = []
    list_answers = []
    for prediction in list_predictions:
        prediction = prediction.replace('\n', ' ')
        fragments = prediction.split('ANSWER:')
        question = fragments[0].replace('QUESTION:', '').strip()
        answer = ''
        if len(fragments) == 2:
            answer = fragments[1].replace('ANSWER:', '').strip()
        list_questions.append(question)
        list_answers.append(answer)
    return list_questions, list_answers
