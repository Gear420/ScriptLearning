import os
file_paths = []
#file_path = "/home/zijian/zxy/gpt/DeScript_LREC2016/esds/pilot_esd/baking a cake.pilot.xml"
def file_name(file_dir):
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.xml':  
                L.append(os.path.join(root, file))  
    return L
l = file_name("/home/zijian/zxy/gpt/DeScript_LREC2016/esds/pilot_esd/")

print(l)


#coding=utf-8
import  xml.dom.minidom
from transformers import * 
import logging
logging.basicConfig(level=logging.INFO)
import torch
#from xml import getElementsByTagName

results = []

for file_path in l[0:1]:
#打开xml文档
    dom = xml.dom.minidom.parse(file_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
#得到文档元素对象
    root = dom.documentElement
#total = 0
#acc = 0
    sentences = []
    import random
    gold_labels = root.getElementsByTagName('script')
#得到文档元素对象
    root = dom.documentElement
#total = 0
#acc = 0
    sentences = []
    import random
    gold_labels = root.getElementsByTagName('script')
    text2 = 'before'
    tokenized_text2 = tokenizer.tokenize(text2)
    indexed_tokens_before = tokenizer.convert_tokens_to_ids(tokenized_text2)
    print(indexed_tokens_before)

    text3 = 'after'
    tokenized_text3 = tokenizer.tokenize(text3)
    indexed_tokens_after = tokenizer.convert_tokens_to_ids(tokenized_text3)
    print(indexed_tokens_after)

    for gold_label in gold_labels:
        sentences = []
    #print(gold_label.getElementsByTagName("item"))
    #randn = random.randint(0 , len(gold_label.getElementsByTagName("item")) - 1 )
    #print(randn)
        sentence = gold_label.getElementsByTagName("item")
        for node  in sentence:
            text = node.getAttribute("original")
            sentences.append(text)
        total = 0
        acc = 0
        if len(sentences) == 1:
            continue
        print(sentences)
        prompt = " [MASK] "
        for i in range(len(sentences)):
            for j in range(i+1,len(sentences)):
                sentence = sentences[i][:-1] + prompt + sentences[j]
                
                print(sentence)
                print(len(sentence))
                tokenized_text = tokenizer.tokenize(sentence)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                segments_ids = [0] * len(tokenized_text)
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segments_ids])
                print(segments_tensors)
                masked_index = tokenized_text.index('[MASK]')
                with torch.no_grad():
                    predictions = model(tokens_tensor)
                #print(indexed_tokens_before)
                #print(masked_index)
                #print(len(predictions[0][0][masked_index]))
                if predictions[0][0][masked_index][indexed_tokens_before] > predictions[0][0][masked_index][indexed_tokens_after]:
                    acc += 1
                total += 1
        print(acc/total)
        results.append(acc/total)

b=len(results)
sum=0
print("数组长度为：",b)
for i in results:
    sum=sum+i
print("均值为：",sum/b)
   		  