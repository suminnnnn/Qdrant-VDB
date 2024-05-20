from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_manager import *
from utils import *

import os

if __name__ == "__main__":
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    os.environ["OPENAI_API_KEY"] ='SECRET KEY'
    
    client = initialize_qdrant_client_in_memory()
    collection_name = "langchain_qdrant_test_collection" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    create_qdrant_collection(client, collection_name, encoder)
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=[" ", ".", ",", "\u200B", "\uff0c", "\u3001", "\uff0e", "\u3002", ""],
        chunk_size=125,
        keep_separator=True,
        is_separator_regex=False,
        chunk_overlap=0
    )
    
    docs = load_and_split_text("./test_file/jpa.txt", splitter)
    
    payload = create_payload(doc_time = datetime.now().strftime("%Y%m%d") ,file_id=1, parent_department_id=1, sub_department_id=2, manager="신수민")
    
    points = create_points(docs, encoder, payload)
    upsert_points(client, collection_name, points)
    
    filter = create_filter("file_id", 1)
    new_docs = load_and_split_text("./test_file/jpa.txt", splitter)
    update_points(client, collection_name, filter, new_docs, encoder, payload)
    
    user_query = "JPA의 영속성 컨텍스트란?"
    query_filter = create_filter(key="manager", value="신수민")
    search_result = search_qdrant(client, collection_name, user_query, encoder, query_filter)
    
    if search_result:
        context_header = f"context: {search_result}"
        user_query = f"\n\nQuery: {user_query}"
        query_constraint = "\n\nConstraint: 질문 Query에 대해서 context에 있는 지식 내에서 정확히 일치하는 경우에만 대답 하고, 모르면 모른다고 대답해."
        prompt = context_header + user_query + query_constraint
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        response = llm.invoke(prompt)
        print("A: " + response.content)
    else:
        print("No search result found.")
