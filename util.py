
import torch
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
import os
from FlagEmbedding import FlagLLMReranker


def get_embeddings_with_model_name(model_name):
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'}, encode_kwargs={'device': 'cpu'})

@staticmethod
def get_embeddings():
    # model_name = "intfloat/multilingual-e5-large"
    # model_name = "../uinetworks_faq/multilingual-e5-large"
    model_name = "/content/multilingual-e5-large"
    embeddings = get_embeddings_with_model_name(model_name)
    return embeddings

# -----------------------------------------------------------------------
embeddings = None
if not embeddings:
    embeddings = get_embeddings()
# -----------------------------------------------------------------------

@staticmethod
def get_pipe():
    # model_name = "HuggingFaceH4/zephyr-7b-beta"
    # model_name = "HuggingFaceH4/zephyr-7b-gemma-v0.1"
    # model_name = "/content/zephyr-7b-gemma-v0.1-coupang"
    # model_name = "/content/zephyr-7b-gemma-v0.1-kr"
    model_name = "/content/zephyr-7b-gemma-v0.1"
    pipe = get_pipe_with_model_name(model_name)
    return pipe


def get_pipe_with_model_name(model_name):
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )
    return pipe


# -----------------------------------------------------------------------
pipe = None
if not pipe:
    pipe = get_pipe()
# -----------------------------------------------------------------------

@staticmethod
def get_reranker():
    reranker = FlagLLMReranker('/content/bge-reranker-v2-gemma', use_fp16=True)
    # reranker = FlagLLMReranker('./bge-reranker-v2-gemma', use_fp16=True)
    return reranker


# -----------------------------------------------------------------------
reranker = None
if not reranker:
    reranker = get_reranker()
# -----------------------------------------------------------------------

"""
    queries_array : [['오늘 점심은 뭐가 맛있을까?', '점심에는 뭐가 맛있을까?'], ['오늘 점심은 뭐가 맛있을까?', '아침 식사는 맛있었니? 222']]
"""
def compute_rerank(queries_array):
    # scores = reranker.compute_score([['오늘 점심은 뭐가 맛있을까?', '점심에는 뭐가 맛있을까?'], ['오늘 점심은 뭐가 맛있을까?', '아침 식사는 맛있었니?'], ['오늘 점심은 뭐가 맛있을까?', '아침 식사는 맛있었니? 111'], ['오늘 점심은 뭐가 맛있을까?', '아침 식사는 맛있었니? 222']])
    scores = reranker.compute_score(queries_array)
    return scores

def get_database():
    # MONGO_URI = "mongodb+srv://ysjeong:jeong7066#@cluster0.jf3wpr7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    MONGO_URI = "mongodb+srv://uinetworks:LKi3dRYprU0NPACI@cluster0.9fjmpd1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    os.environ["MONGO_URI"] = MONGO_URI
    DB_NAME = "nodong_qa"

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db_nodong_qa = client[DB_NAME]
    return db_nodong_qa

def get_db_table(table_name):
    db_nodong_qa = get_database()
    db_table = db_nodong_qa[table_name]
    return db_table

def get_nodong_qa():
    COLLECTION_NAME_QA = "nodong_qa"
    return get_db_table(COLLECTION_NAME_QA)

def get_nodong_qa_content():
    COLLECTION_NAME_QA_CONTENT = "nodong_qa_content"
    return get_db_table(COLLECTION_NAME_QA_CONTENT)


# -----------------------------------------------------------------------
# db_coupang_faq = get_database()
# faq_doc = get_faq_doc()
# faq_qa = get_faq_qa()
# -----------------------------------------------------------------------


# get file name
def get_file_name(f_path):
    split_char = "/"
    if "/" not in f_path:
        split_char = "\\"

    f_name = f_path.split(split_char)[-1].replace(".pdf", "").strip()
    return f_name.strip()


"""
  make_prompt
"""
def make_prompt_of_qas_list(query, qas_list):

    qas_text_arr = []
    for idx, qa in enumerate(qas_list):
        q = qas_list[idx]["question"]
        a = qas_list[idx]["answer"]
        idx += 1
        # qas_text_arr.append(f"{idx}.Question: {q}\n A: {a}\n")
        qas_text_arr.append(f"{idx}.Question: \n{q}\n")
        qas_text_arr.append(f"{idx}.Answer: \n{a}\n\n")
    qas_text = "".join(qas_text_arr)

    prompt = f"""
          You are a Inquiry answer bot. You will be given some Question and Answer sets.
          Your task is to generate the appropriate reply for Inquery based on Question and Answer sets.
          Inquery after <<< >>> :

          ####
          Question and Answer sets:

{qas_text}
          ####

          You can reference Question and Answer sets to generate the reply to Inquiry.
          The reply text content should be within the Answer of Question and Answer sets.
          Do not say that you can not reply.
          Do not refer about Question and Answer sets itself including the No of Question and Answer sets.
          You must always reply in Korean.
          Your answer should be logical and make sense.
          If There are related information (관련 정보), respond it as much as possible.
          If there are href link, keep href without any changes.

          <<<
        Inquiry: {query}
          >>>

    """

    return prompt


"""
  get_answer_by_llm 실행
  기능 : qas_list (Q/A list)를 참조하여 query에 적합한 답변을 생성한다.
"""
def get_answer_by_llm(query, qas_list):
    prompt = make_prompt_of_qas_list(query, qas_list)
    messages = [
        {
            "role": "system",
            "content": "",  # Model not yet trained for follow this
        },
        {"role": "user", "content": prompt},
    ]
    max_new_tokens = 1024
    outputs = pipe(
        messages,
        # max_new_tokens=128,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        stop_sequence="<|im_end|>",
    )
    print(outputs[0]["generated_text"][-1]["content"])

    answer = outputs[0]["generated_text"][-1]["content"]

    return answer


"""
    compute_result : rerank result
    child_qas_list : embedding result
"""
def get_sorted_qas_list(query, child_qas_list):

    # 중복제거
    q_list = []
    qa_temp = []
    for qas in child_qas_list:
        if qas["question"] in qa_temp:
            continue
        qa_temp.append(qas["question"])
        q_list.append(qas)

    child_qas_list = q_list

    print("-----------------------------")
    queries_array = []
    queries_eval_array = []
    print("query:{}".format(query))
    print("--------------")
    for qa in child_qas_list:
        queries_array.append([query, qa["question"]])
        queries_eval_array.append(qa["score"])
        print(qa["question"])
    print("-----------------------------")

    # Rerank child qas list
    compute_result = compute_rerank(queries_array)

    # TODO : 삭제
    print(queries_eval_array)
    print(compute_result)

    sorted_result = list(reversed(sorted((e, i) for i,e in enumerate(compute_result))))
    r_qas_list = []
    t_cnt1 = 0
    t_cnt2 = 0
    t_cnt3 = 0
    for idx, cr in enumerate(sorted_result):
        print("cr:{}".format(cr[1]))
        if idx == 0:
            t_cnt1 = len(child_qas_list[cr[1]]["answer"])
        if idx == 1:
            t_cnt2 = len(child_qas_list[cr[1]]["answer"])
        if idx == 2:
            t_cnt3 = len(child_qas_list[cr[1]]["answer"])
        r_qas_list.append(child_qas_list[cr[1]])

    return r_qas_list, t_cnt1, t_cnt2, t_cnt3


"""
    get query_member_cnt
    : query_member_cnt 를 return 한다.
"""
def get_query_member_cnt(default_member_cnt, t_cnt1, t_cnt2, t_cnt3):
    # content 길이에 따라 query_member_cnt 을 조정한다.
    t_cnt = t_cnt1 + t_cnt2 + t_cnt3

    query_member_cnt = default_member_cnt
    t_cnt_limit = 6000
    if t_cnt > t_cnt_limit:
        query_member_cnt = 2
    if (t_cnt1 + t_cnt2) > t_cnt_limit:
        query_member_cnt = 1

    return query_member_cnt


"""
  get_answer_by_embedding
"""
def get_answer_by_embedding(embeddings, nodong_qa, query):

    query_embedding = embeddings.embed_documents([query.strip()])[0]

    # Retrieve relevant child documents based on query
    nodong_qas = nodong_qa.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding_q",
                "queryVector": query_embedding,
                "numCandidates": 7,
                "limit": 7
            }
        },
        {
            "$project": {
                "category": 1,
                "url": 1,
                "question": 1,
                "answer": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ])

    nodong_qas_list = list(nodong_qas)

    # merge nodong qa content search result
    nodong_qas_list += get_search_qa_by_content(query)

    # compute 결과와 child_qas_list 결과를 조합하여 qas list 구성
    nodong_qas_list, t_cnt1, t_cnt2, t_cnt3 = get_sorted_qas_list(query, nodong_qas_list)
    print("t_cnt1/t_cnt2/t_cnt3:{}/{}/{}".format(t_cnt1, t_cnt2, t_cnt3))

    query_member_cnt = 3
    query_member_cnt = get_query_member_cnt(query_member_cnt, t_cnt1, t_cnt2, t_cnt3)
    if len(nodong_qas_list) > query_member_cnt:
        nodong_qas_list = nodong_qas_list[:query_member_cnt]
        # TODO : 화인 후 적용
        """
        if query_member_cnt == 1: # 6000 byte 초과하는 경우 대응
            if len(nodong_qas_list[0]["answer"]) > 6000:
                nodong_qas_list[0]["answer"] = nodong_qas_list[0]["answer"][:6000]
        """

    print("The length of child_qas_list:{}".format(len(nodong_qas_list)))

    answer = ""
    if len(nodong_qas_list) > 0:
        print(nodong_qas_list[0]["question"])
        print("========================================")
        print(nodong_qas_list[0]["answer"])
        print("========================================")
        print(nodong_qas_list[0]["score"])

    question = nodong_qas_list[0]["question"]
    answer = nodong_qas_list[0]["answer"]
    url = nodong_qas_list[0]["url"]
    score = float(nodong_qas_list[0]["score"])

    return question, answer, url, score, nodong_qas_list


def get_search_qa_by_content(query):
    # nodong table search
    nodong_qa = get_nodong_qa()
    nodong_qa_content = get_nodong_qa_content()

    # query embedding
    query_embedding = embeddings.embed_documents([query.strip()])[0]

    # Retrieve relevant child documents based on query
    content_qas = nodong_qa_content.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 3,
                "limit": 3
            }
        },
        {
            "$project": {
                "url": 1,
                "content": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ])

    content_qas_list = list(content_qas)

    for c in content_qas_list:
        print(c["content"])
        print(c["url"])
        print(c["score"])

    qa_list = None

    # content url 목록으로 nodong qa 조회
    if len(content_qas_list) > 0:

        # nodong qa 조회
        qa_finds = nodong_qa.find({
            "url": {"$in": [c["url"] for c in content_qas_list]}
        })

        qa_list = list(qa_finds)

        # TODO : 확인 후 삭제
        print("--------the start of get_search_qa_by_content----------")
        for idx, c in enumerate(qa_list):
            # print(c["url"])
            # print(c["answer"])
            content = content_qas_list[idx]["content"]
            c["score"] = content_qas_list[idx]["score"]
            print(c["question"])
            print(content)
            print(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
        print("--------the end of get_search_qa_by_content----------")

    return qa_list


def querying(query, history):

    nodong_qa = get_nodong_qa()

    process_type = "LLM"
    answer = ""
    score = -1
    llm_answer = ""
    process_type = "Embedding"

    # search by embedding
    question, answer, url, score, qas_list = get_answer_by_embedding(embeddings, nodong_qa, query)

    if score < 0.97:
        llm_answer = get_answer_by_llm(query, qas_list)

    return_text_arr = []
    return_text_arr.append(f"<h2>Process type</h2>\n{process_type}")
    return_text_arr.append(f"<h2>Question</h2>\n{question}")
    return_text_arr.append(f"<h2>Answer</h2>\n{answer}")
    return_text_arr.append(f"<h2>Link URL</h2>\n<a href='{url}'>{url}</a>")
    if process_type == "Embedding":
        return_text_arr.append(f"<h2>Score</h2>\n{score}")
    if len(llm_answer) > 0:
        return_text_arr.append(f"<h2>LLM answer</h2>\n{llm_answer}")
    return_text = "".join(return_text_arr)

    return return_text

# TODO : 삭제
# query = "취업규칙이 변경되면 기존 근로계약과의 우선순위는 어떻게 달라지게 되나요?"
# querying(query, None)
