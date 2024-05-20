from datetime import datetime
from langchain_community.document_loaders import TextLoader
from qdrant_client.http.models import models


def create_filter(key, value):
    return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))])

def load_and_split_text(file_path, splitter, encoding="UTF-8"):
    loader = TextLoader(file_path, encoding=encoding)
    data = loader.load()
    return splitter.split_text(data[0].page_content)

def create_payload(doc_time,file_id, parent_department_id, sub_department_id, manager):
    upload_time = datetime.now().strftime("%Y%m%d")
    return {
        "doc_time": doc_time,
        "upload_time": upload_time,
        "file_id": file_id,
        "parent_department_id": parent_department_id,
        "sub_department_id": sub_department_id,
        "manager": manager,
    }
