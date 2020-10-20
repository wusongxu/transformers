from transformers import BertTokenizer, BertModel,BertForMaskedLM
import torch


# if __name__ == '__main__':
#     tokenizer = BertTokenizer.from_pretrained('/datadisk2/songxu/output/checkpoint-100000')
#     model = BertModel.from_pretrained('/datadisk2/songxu/output/checkpoint-100000')
#     batch_sentenct = ['在吗？', '是的', '保质期多久', '什么时候发货', '今天发货吗', '发什么快递']
#     input_id = tokenizer(batch_sentenct, padding=True, truncation=True, max_length=12, return_tensors='pt')
#     # print(input_id)
#     # print(input_id['input_ids'].shape)
#
#     output = model(input_id['input_ids'])
#     print(output[0].shape)
#     print(output[1].shape)
#     for i in range(output[0].shape[0]):
#         if i+1< output[0].shape[0]:
#             import pdb;pdb.set_trace()
#             test_1 = torch.sum(output[0][i],dim=1)
#             test_2 = torch.sum(output[0][i+1],dim=1)
#             cosine = torch.cosine_similarity(test_1,test_2,dim=1)
#             # print(cosine)
#             print(cosine)

tokenizer = BertTokenizer.from_pretrained('/datadisk2/songxu/output/checkpoint-100000')
model = BertForMaskedLM.from_pretrained('/datadisk2/songxu/output/checkpoint-100000')

sentence = "9月22日下午，习近平总书记主持召开教育文化卫生体育领域专家代表座谈会并发表重要讲话，就“十四五”时期经济社会发展听取意见和建议。习近平总书记强调，党和国家高度重视教育、文化、卫生、体育事业发展，党的十八大以来党中央就此作出一系列战略部署，各级党委和政府要抓好落实工作，努力培养担当民族复兴大任的时代新人，扎实推进社会主义文化建设，大力发展卫生健康事业，加快体育强国建设，推动各项社会事业增添新动力、开创新局面，不断增强人民群众获得感、幸福感、安全感"
tokens = ['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]']

for i in range(1, len(tokens)-1):
    tmp = tokens[:i] + ['[MASK]'] + tokens[i+1:]

    masked_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tmp)])
    segment_ids = torch.tensor([[0]*len(tmp)])

    outputs = model(masked_ids, token_type_ids=segment_ids)
    prediction_scores = outputs[0]
    print(tmp)
    # 打印被预测的字符
    prediction_index = torch.argmax(prediction_scores[0, i]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([prediction_index])[0]
    print(predicted_token)







