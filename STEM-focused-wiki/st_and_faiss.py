from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, load_from_disk
import faiss
from faiss import write_index, read_index

## sentence-transformers/all-MiniLM-L6-v2 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

print("successful loading tokenizer and model\n")

## sentence-transformers/all-MiniLM-L6-v2 모델 저장하기
tokenizer.save_pretrained("output/all-MiniLM-L6-v2")
model.save_pretrained("output/all-MiniLM-L6-v2")

print("successful saving tokenizer and model\n")

## wiki dataset 불러오기
paraphs_parsed_dataset = load_from_disk("dataset/all-paraphs-parsed-expanded")
modified_texts = paraphs_parsed_dataset.map(lambda example:
                                             {'temp_text':
                                              f"{example['title']} {example['section']} {example['text']}".replace('\n'," ").replace("'","")},
                                             num_proc=2)["temp_text"]

print("successful loading wikidataset\n")

## faiss index 만들기
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
vectors = encoder.encode(modified_texts)
vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)
write_index(index, "output/all-MiniLM-L6-v2/faiss.index")

print("successful writing faiss index\n")
