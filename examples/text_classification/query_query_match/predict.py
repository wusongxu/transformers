import sys
sys.path.append(r'.')
from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('/datadisk2/songxu/qq_match/checkpoint-2000')
model = BertForSequenceClassification.from_pretrained('/datadisk2/songxu/qq_match/checkpoint-2000')


def split_input_to_batch(inputs_list, batch_size):
    all_batch = []
    num_batch = int(len(inputs_list) / batch_size)
    for i in range(num_batch):
        all_batch.append(inputs_list[i * batch_size:(i + 1) * batch_size])
    last_data = inputs_list[int(len(inputs_list) / batch_size) * batch_size:]
    if last_data:
        all_batch.append(last_data)
    return all_batch


def predict_one_batch(batch_inputs):
    batch_ids = []
    batch_segment_ids = []
    batch_attention_mask = []
    for line in batch_inputs:
        sentence_a = line['sentence_a']
        sentence_b = line['sentence_b']
        token_sentence_a = tokenizer.tokenize(sentence_a)
        token_sentence_b = tokenizer.tokenize(sentence_b)
        if len(token_sentence_a) >= 62 or len(token_sentence_b) >= 62:
            token_sentence_a = token_sentence_a[:62]
            token_sentence_b = token_sentence_b[:62]
        tokens = ['[CLS]'] + token_sentence_a + ['[SEP]'] + token_sentence_b + ['[SEP]']
        if len(tokens) >= 128:
            padding_length = 0
        else:
            padding_length = 128 - len(tokens)
        batch_ids.append(tokenizer.convert_tokens_to_ids(tokens) + [0] * (padding_length))
        batch_attention_mask.append([1] * len(tokens) + [0] * padding_length)
        batch_segment_ids.append(
            [0] * (len(token_sentence_a) + 2) + [1] * (len(token_sentence_b) + 1) + [0] * padding_length)

    input_ids = torch.tensor(batch_ids)
    segment_ids = torch.tensor(batch_segment_ids)
    attention_mask = torch.tensor(batch_attention_mask)
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        segment_ids = segment_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        model.to("cuda")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        predict_label = torch.argmax(outputs[0], dim=-1).cpu().numpy().tolist()

    for i, line in enumerate(batch_inputs):
        if predict_label[i] == 0:
            batch_inputs[i]['label'] = '否'
            batch_inputs[i]['proba'] = float(outputs[0][i][0])
        else:
            batch_inputs[i]['label'] = '是'
            batch_inputs[i]['proba'] = float(outputs[0][i][1])
    return batch_inputs


def predict_batch(inputs_list, batch_size):
    result = []
    all_batch = split_input_to_batch(inputs_list, batch_size)
    for batch in all_batch:
        batch = predict_one_batch(batch)
        result += batch
    return result


if __name__ == '__main__':
    data = [{"sentence_a": "怎么辨别真假", "sentence_b": "和田玉怎么鉴定真假"},
            {"sentence_a": "现在买什么时候出票", "sentence_b": "还没收到货已经确认收货了点错了"}
            ]
    result = predict_batch(data, 20)
    print(result)
