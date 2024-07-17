from nltk.tokenize import sent_tokenize
import re
import nltk
import pandas as pd
import tiktoken

class SentenceSplitter:
    def __init__(self, chunk_size: int):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        self.chunk_size = chunk_size
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    

    def tokens(self, text) -> int:
        return len(self.encoding.encode(text))

    def get_last_sentence(self, text: str) -> str:
        sentences = sent_tokenize(text)
        return sentences 

    def create_sentence_df(self, text: str) -> pd.DataFrame:
        sentences = self.get_last_sentence(text)
        df = pd.DataFrame(sentences, columns=['sentence'])
        df['tokens'] = df['sentence'].apply(self.tokens)
        return df

    def split_long_sentence(self, sentence: str) -> list:
        breakpoints = [';', ',', ' and ', ' but ', ' or ', ' however ', ' therefore ']
        parts = [sentence]

        for bp in breakpoints:
            new_parts = []
            for part in parts:
                if self.tokens(part) > self.chunk_size:
                    new_parts.extend(re.split(r'(?<={}) '.format(re.escape(bp)), part))
                else:
                    new_parts.append(part)
            parts = new_parts

        return [part for part in parts if part and self.tokens(part) <= self.chunk_size]

    def _merge_parts(self, sentences: list) -> list:
        complete_sentences = []
        current_chunk = ''
        current_tokens = 0

        for sentence in sentences:
            num_tokens = self.tokens(sentence)
            if current_tokens + num_tokens <= self.chunk_size:
                current_chunk += (' ' + sentence).strip()
                current_tokens += num_tokens
            else:
                complete_sentences.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = num_tokens

        if current_chunk:
            complete_sentences.append(current_chunk.strip())

        return complete_sentences

    def semantic_chunking(self, text: str) -> pd.DataFrame:
        df = self.create_sentence_df(text)
        complete_sentences = []
        current_chunk = ''
        current_tokens = 0

        for _, row in df.iterrows():
            sentence = row['sentence']
            num_tokens = row['tokens']

            # If sentence is longer than the chunk size, split and merge it separately
            if num_tokens > self.chunk_size:
                split_sentences = self.split_long_sentence(sentence)
                merged_sentences = self._merge_parts(split_sentences)
                complete_sentences.extend(merged_sentences)
            else:
                # Accumulate sentences until the current chunk reaches or exceeds the chunk size
                if current_tokens + num_tokens <= self.chunk_size:
                    current_chunk += (' ' + sentence).strip()
                    current_tokens += num_tokens
                else:
                    complete_sentences.append(current_chunk.strip())
                    current_chunk = sentence
                    current_tokens = num_tokens

        # Add the last chunk if it's not empty
        if current_chunk:
            complete_sentences.append(current_chunk.strip())

        semantic_df = pd.DataFrame(complete_sentences, columns=['chunked_sentence'])
        semantic_df['tokens'] = semantic_df['chunked_sentence'].apply(self.tokens)
        return semantic_df