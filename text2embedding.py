import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained("bert-large-uncased")
import csv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
print('the model has been loaded to GPU')

# 2. 加载您的数据，包括标题和摘要
data = pd.read_excel('preprocessed_data.xlsx')  # 替换为您的数据文件路径
data['Abstract'] = data['Abstract'].apply(eval)

def concat_word(text_list):
    # 将字符列表转换为字符串
    text = ' '.join(text_list)
    return text

data['Abstract'] = data['Abstract'].apply(concat_word)
# 3. 定义一个函数来提取特征向量并计算平均特征向量
def extract_features(text_lists):
    # 将字符列表转换为字符串
    inputs = tokenizer(text_lists, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    return embeddings

word_vectors = {}

BATCH_SIZE = 64
# 6. 遍历每个摘要并分词，然后将每个词的特征向量添加到字典中
print(f'start calculation embedding')
for i in tqdm(range(0, len(data['Abstract']), BATCH_SIZE)):
    batch_abstracts = data['Abstract'][i:i + BATCH_SIZE].tolist()
    batch_vectors = extract_features(batch_abstracts)

    for abstract, vectors in zip(batch_abstracts, batch_vectors):
        tokens = tokenizer.tokenize(''.join(abstract))
        
        word = ''
        word_vector = None
        
        for token, vector in zip(tokens, vectors):
            
            token = token.lower()
            if token.startswith('##'):
                # 如果token是单词的一部分，累加其嵌入
                word += token[2:]
                word_vector += vector
            else:
                if word:
                    if word not in word_vectors:
                        word_vectors[word] = []
                    word_vectors[word].append(word_vector.cpu())
                word = token
                word_vector = vector

print(f'calculate average embedding for each word')
average_word_vectors = {}
for word, vectors in tqdm(word_vectors.items(), desc="cal average embed"):
    average_vector = torch.stack(vectors).mean(dim=0)
    average_word_vectors[word] = average_vector.numpy()

# 8. 存储结果到Excel文件
with open('bert_word_vectors.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Word', 'Vector'])
    for word, vector in average_word_vectors.items():
        writer.writerow([word] + list(vector))