import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
from rank_bm25 import BM25Okapi

load_dotenv()


class SearchEngine:
    def __init__(self, milvus_client, milvus_collection_name):
      self.milvus_client = milvus_client
      self.milvus_collection_name = milvus_collection_name
      self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def tokenize(self, text):
        return text.lower().split()
    
    def rerank_with_bert(self, knowledge_base_texts, query_embedding):
        similarity_scores = np.dot(self.model.encode(knowledge_base_texts), query_embedding)
        reranked_indices = np.argsort(similarity_scores)[::-1]  # descending order
        return reranked_indices
  
    def query_milvus(self, embedding):
        result_count = 10
  
        result = self.milvus_client.search(
            collection_name=self.milvus_collection_name,
            data=[embedding],
            limit=result_count,
            output_fields=["path", "text"])
  
        list_of_knowledge_base = list(map(lambda match: match['entity']['text'], result[0]))
        list_of_sources = list(map(lambda match: match['entity']['path'], result[0]))
  
        return {
            'list_of_knowledge_base': list_of_knowledge_base,
            'list_of_sources': list_of_sources
        }
    
    def retrieve_and_rerank(self, user_query):
        query_embedding = self.model.encode(user_query)

        knowledge_base_texts, sources = self.query_milvus(query_embedding)

        bm25_corpus = [self.tokenize(doc) for doc in knowledge_base_texts]
        bm25 = BM25Okapi(bm25_corpus)
        tokenized_query = self.tokenize(user_query)
        bm25_scores = bm25.get_scores(tokenized_query)

        reranked_indices = self.rerank_with_bert(knowledge_base_texts, query_embedding)

        reranked_knowledge_base = [knowledge_base_texts[idx] for idx in reranked_indices]
        reranked_sources = [sources[idx] for idx in reranked_indices]

        return reranked_knowledge_base, reranked_sources
  
    def query_vector_db(self, embedding):
        return self.query_milvus(embedding)
  
    def ask_groq(self, knowledge_base, user_query):
        system_content = """You are an AI coding assistant designed to help users with their programming needs based on the Knowledge Base provided.
        If you dont know the answer, say that you dont know the answer.
        Only answer questions using data from knowledge base and nothing else.
        """
  
        user_content = f"""
            Knowledge Base:
            ---
            {knowledge_base}
            ---
            User Query: {user_query}
            Answer:
        """
        system_message = {"role": "system", "content": system_content}
        user_message = {"role": "user", "content": user_content}

        client = Groq(
            api_key=os.getenv('GROQ_API'),
        )
        
        response = client.chat.completions.create(messages=[system_message, user_message],model="llama3-8b-8192")
        return response.choices[0].message.content
  
    def search(self, user_query):
        reranked_knowledge_base, reranked_sources = self.retrieve_and_rerank(user_query)
        knowledge_base = "\n".join(reranked_knowledge_base)

        return {
            'sources': reranked_sources,
            'response': self.ask_groq(knowledge_base, user_query)
        }