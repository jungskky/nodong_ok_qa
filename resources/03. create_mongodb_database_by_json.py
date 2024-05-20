
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import json
import os

# ##################################################
# Embeddings settings
# ##################################################
def get_embeddings():
    encoding_path = "C:/PycharmProjects/uinetworks_faq/multilingual-e5-large"
    embeddings = HuggingFaceEmbeddings(model_name=encoding_path, model_kwargs={'device': 'cpu'},
                                       encode_kwargs={'device': 'cpu'})
    return embeddings

embeddings = get_embeddings()

# ##################################################
# MongoDB settings
# ##################################################
def get_mongodb():
    MONGO_URI = "mongodb+srv://ysjeong:jeong7066#@cluster0.jf3wpr7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    os.environ["MONGO_URI"] = MONGO_URI
    # DB_NAME = "nodong_qa"
    # COLLECTION_NAME_QA = "nodong_qa_test"

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    return client

def get_nodong_database():
    client = get_mongodb()
    DB_NAME = "nodong_qa"
    db_nodong_qa = client[DB_NAME]
    return db_nodong_qa

def get_nodong_qa():
    db_nodong_qa = get_nodong_database()
    # test 후 삭제
    # COLLECTION_NAME_QA = "nodong_qa"
    COLLECTION_NAME_QA = "nodong_qa"
    nodong_qa = db_nodong_qa[COLLECTION_NAME_QA]
    return nodong_qa

nodong_qa = get_nodong_qa()

# ##################################################
# Execute function
# ##################################################

def make_nodong_qa_by_json():
    # file_path = "./nodong_qa_list.json"
    # file_path = "./nodong_qa_list_best_qa.json"
    # file_path = "./nodong_qa_list_sector.json"
    # file_path = "./nodong_qa_list_admin_examples.json"
    file_path = "./nodong_qa_list_nodong_examples.json"
    with open(file_path, "r", encoding="utf-8") as f:
        nodong_data = json.load(f)
        for d in nodong_data:
            print(d["url"])
            print(d["question"])
            print(d["answer"])
            url = d["url"]
            question = d["question"]
            answer = d["answer"]

            # MongoDB insert
            question_embedding = embeddings.embed_documents([question])[0]
            nodong_qa.insert_one({
                'url': url,
                'question': question,
                'embedding_q': question_embedding,
                'answer': answer
            })

# execute
make_nodong_qa_by_json()
