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

# Point Create

points = []

for idx, doc in enumerate(docs):
    
    print(f'{idx=}')
    print(f'{doc=}')

    print(f'{len(doc)=}')
    print()
    
    embedding = encoder.encode(doc)
        
    point = PointStruct(
        id = ((idx +1) * pow(21, idx)) % 1234567891 ,
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

# Point Delete, Update, Scroll에 사용할 Filter 정의

## file_id 값을 기준으로 검색
filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key='file_id',
                                match=models.MatchValue(value=1),
                            ),],
                    )

point_selector = models.FilterSelector(filter=filter)

scroll_filter = models.Filter(
    must=[
        models.FieldCondition(key="file_id", match=models.MatchValue(value=1))
    ]
)

# Point Delete

# 삭제하고자 하는 file_id를 가지는 Point 들의 Point id 조회
points = client.scroll(collection_name=collection_name, scroll_filter=scroll_filter)[0]

point_ids = [point.id for point in points]

# vector 삭제
client.delete_vectors(collection_name=collection_name, wait=True, points=point_ids, vectors=[''])

# point 삭제
client.delete(collection_name=collection_name, wait=True, points_selector=point_ids)

# Point Update

point_selector = models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key='file_id',
                                match=models.MatchValue(value=1),
                            ),
                        ],
                    )
                )

scroll_filter = models.Filter(
    must=[
        models.FieldCondition(key="file_id", match=models.MatchValue(value=1))
    ]
)

# 업데이트 할 파일의 point 조회

points = client.scroll(collection_name=collection_name, scroll_filter=scroll_filter)[0]

point_ids = [point.id for point in points]

# 기존 Point 삭제

client.delete_vectors(collection_name=collection_name, wait=True, points=point_ids, vectors=[''])
client.delete(collection_name=collection_name, wait=True, points_selector=point_ids)

# 새로운 Point 저장

loader = TextLoader("./AboutDocker.txt", encoding="UTF-8")
data = loader.load()
docs = text_splitter.split_text(data[0].page_content)

points = []

for idx, doc in enumerate(docs):
    
    print(f'{idx=}')
    print(f'{doc=}')

    print(f'{len(doc)=}')
    print()
    vector = encoder.encode(doc)
        
    point = models.PointStruct(
        id = ((idx +2) * pow(31, idx)) % 1234567891 ,
        vector=vector,
        payload={
            "doc_time" : datetime(2022, 2, 21).strftime("%Y%m%d"),
            "upload_time" : datetime.now().strftime("%Y%m%d"),
            "file_id" : 3,
            "parent_department_id" : 2,
            "sub_department_id" : 1,
            "manager" : "관리자",
            "doc" : doc
        }
    )
    
    points.append(point)

client.upsert(collection_name=collection_name, points=points)

# Point Retrieve

user_query = "JPA의 영속성 컨텍스트란?"

query_vector = encoder.encode(user_query).tolist()
query_filter = models.Filter(must=[models.FieldCondition(key="manager", match=models.MatchValue(value="관리자"))])

hits = client.search(collection_name=collection_name,
                     query_vector= query_vector, query_filter=query_filter, limit=2)

search_result = hits[1].payload['doc']

context_header = f"context: {search_result}"

user_query = f"\n\nQuery: {user_query}"

query_constraint = "\n\nConstraint: 질문 Query에 대해서 context에 있는 지식 내에서 정확히 일치하는 경우에만 대답 하고, 모르면 모른다고 대답해."

prompt = context_header + user_query + query_constraint


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

response = llm.invoke(prompt)

print("A: " + response.content)