from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from datetime import datetime


model = SentenceTransformer('all-MiniLM-L6-v2')

# Qdrant 클라이언트 초기화 (수정 필요)
client = QdrantClient(host="localhost", port=6333)
#client = QdrantClient(host="talkable2-vdb", port=6333)

collection_name = "langchain_qdrant_test_collection"

# Qdrant Collection 생성
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="created_at",
    field_schema="datetime"
)

# 임베딩할 파일 로드

loader = TextLoader("./jpa.txt", encoding="UTF-8")
data = loader.load()
    
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap  = 100,
    length_function = len,
)

docs = text_splitter.split_text(data[0].page_content)

# 각 문서를 Point로 임베딩, 저장

for idx, doc in enumerate(docs):
    
    embedding = model.encode(doc)
        
    point = PointStruct(
        id = idx,
        vector=embedding,
        created_at=datetime.now().date()
    )

    client.upsert(collection_name=collection_name, points=[point])

print("Data upload to Qdrant")