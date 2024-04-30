from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, VectorParams, Distance

import tiktoken
from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


from datetime import datetime
import os


encoder = SentenceTransformer("all-MiniLM-L6-v2")

os.environ["OPENAI_API_KEY"] ='SECRET-KEY'

# Qdrant 클라이언트 초기화
client = QdrantClient(host="localhost", port=6333)

# Qdrant Collection 생성
collection_name = "langchain_qdrant_test_collection"

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE)
)

# 임베딩할 텍스트 파일 로드

loader = TextLoader("./jpa.txt", encoding="UTF-8")
data = loader.load()

# 토큰 수를 기준으로 청크로 분리
# RecursiveCharacterTextSplitter: 단락 -> 문장 -> 단어 순서로 재귀적으로 분할

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    
    # 청크 구분자 설정
    separators=[
        " ",
        ".",
        ",",
        "\u200B",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        ""
    ],
    
    # 한 청크에 포함될 최대 토큰 수
    chunk_size = 125,
    
    # 구분자를 청크의 일부로 유지할지 여부 설정
    keep_separator = True,
    
    # 구분자로 정규식을 사용할지 여부 설정
    is_separator_regex=False,

    # 청크 간의 중복되는 문자 수 설정
    chunk_overlap  = 0
)

docs = text_splitter.split_text(data[0].page_content)

# 각 청크를 임베딩 후 Point로 저장

points = []

for idx, doc in enumerate(docs):
    
    print(f'{idx=}')
    print(f'{doc=}')

    print(f'{len(doc)=}')
    print()
    embedding = model.encode(doc)
        
    point = PointStruct(
        id = idx,
        vector=embedding,
        payload={
            "doc_time" : datetime(2022, 2, 21).strftime("%Y%m%d"),
            "upload_time" : datetime.now().strftime("%Y%m%d"),
            "file_id" : 2,
            "parent_department" : "소프트웨어융합대학",
            "sub_department" : "소프트웨어학과",
            "manager" : "신수민",
            "doc" : doc
        }
    )
    
    points.append(point)

client.upsert(collection_name=collection_name, points=points)

print("Data upload to Qdrant")