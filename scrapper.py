import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from semantic_chunk_splitter import SentenceSplitter
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import MilvusClient
import re

connections.connect(host='localhost', port='19530')
client = MilvusClient(host='localhost', port='19530')

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=10000)
]
schema = CollectionSchema(fields, "Collection of text chunks with metadata")
collection_name = "nvidia_dataset_6"


if collection_name not in client.list_collections():
    client.create_collection(
        collection_name=collection_name,
        dimension=384,  # Ensure the dimension matches your schema
        schema=schema
    )

print("Collection created or already exists")

collection = Collection(collection_name)

index_params = {
    "index_type": "IVF_FLAT",  
    "metric_type": "L2",       
    "params": {"nlist": 128}  
}
collection.create_index(field_name="vector", index_params=index_params)
print("Index created")

collection.load()
print("Collection loaded")

class Indexer:
    def __init__(self, milvus_client, milvus_collection_name):
        self.link_count = 0
        self.max_links = 30
        self.milvus_client = milvus_client
        self.milvus_collection_name = milvus_collection_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.visited_links = set()

    def get_html_sitemap(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        links = []
        # base_url = re.match(r'(.*//.*?)/', url).group(1)
        a_tags = soup.find_all("a")
        for tag in a_tags:
            href = tag.get("href")
            if href.startswith('https://'):
                links.append(href)
            elif href.startswith('#'):
                continue  
            else:
                href = url + href
                links.append(href)
        # print(links)
        return links

    def get_html_body_content(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        body = soup.body
        inner_text = body.get_text() if body else ''
        return inner_text

    def index_website(self, website_url, depth=1, max_depth=5):
        if depth > max_depth or self.link_count >= self.max_links:
            return
        
        if website_url in self.visited_links:
            return
        self.visited_links.add(website_url)

        links = self.get_html_sitemap(website_url)
        
        for link in links:
            if self.link_count >= self.max_links:
                break
            
            if link not in self.visited_links:
                try:
                    content = self.get_html_body_content(link)
                    self.add_html_to_vectordb(content, link)
                    self.visited_links.add(link)
                    self.link_count += 1
                    print(link) 
                    self.index_website(link, depth + 1, max_depth)
                except requests.RequestException as e:
                    print(f"Request error: {e}")
                except AttributeError as e:
                    print(f"Attribute error: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

    def add_html_to_vectordb(self, content, path):
        text_splitter = SentenceSplitter(chunk_size=600)
        docs = text_splitter.semantic_chunking(content)
        for index, row in docs.iterrows():
            embedding = self.model.encode(row['chunked_sentence'])
            self.insert_embedding(embedding, row['chunked_sentence'], path)

    def insert_embedding(self, embedding, text, path):
        entity = {
            "vector": embedding.tolist(),
            "text": text,
            "path": path
        }
        self.milvus_client.insert(self.milvus_collection_name, [entity])

indexer = Indexer(client, milvus_collection_name=collection_name)
indexer.index_website('https://docs.nvidia.com/cuda/')
# indexer.get_html_sitemap('https://docs.nvidia.com/cuda/')

num_entities = collection.num_entities
print(f"Number of entities in the collection: {num_entities}")
