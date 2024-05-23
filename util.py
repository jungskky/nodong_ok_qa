
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
    MONGO_URI = "mongodb+srv://ysjeong:jeong7066#@cluster0.jf3wpr7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    os.environ["MONGO_URI"] = MONGO_URI
    DB_NAME = "nodong_qa"

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db_coupang_faq = client[DB_NAME]
    return db_coupang_faq

def get_db_table(table_name):
    db_coupang_faq = get_database()
    db_table = db_coupang_faq[table_name]
    return db_table

def get_nodong_qa():
    COLLECTION_NAME_QA = "nodong_qa"
    return get_db_table(COLLECTION_NAME_QA)

# def get_faq_doc():
#     COLLECTION_NAME_DOC = "faq_doc"
#     return get_db_table(COLLECTION_NAME_DOC)
#
# def get_faq_qa():
#     COLLECTION_NAME_QA = "faq_qa"
#     return get_db_table(COLLECTION_NAME_QA)


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
def make_prompt(query, query_list_txt, inquiry_examples_txt):
    prompt = f"""
          You are a query maker bot. Your task is to choose only one query below
          and choose most match query after <<< >>> into one of the following predefined query list:

          ####
          query list:

{query_list_txt}
          ####


          If the Inquiry doesn't fit into any of the above query list, classify it as:
          not matched

          You will only respond with the predefined query list.
          Don't provide additional explanations or text.
          You must reply with one of the query list without any change.
          Don't add additional comment or text.


          ####
          Here are some Inquiry / query examples:

{inquiry_examples_txt}
          ####

          <<<
        Inquiry: {query}
          >>>

    """
    return prompt


"""
  get_category_by_query
"""
def get_category_by_query(embeddings, faq_doc, query):

    query_embedding = embeddings.embed_documents([query.strip()])[0]

    # print("query_embedding: {}".format(query_embedding))

    # Retrieve relevant child documents based on query
    child_docs = faq_doc.aggregate([{
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 10,
            "limit": 1
        }
    }])

    child_docs_list = list(child_docs)
    # print("The length of list [{}]".format(len(child_docs_list)))

    if len(child_docs_list) > 0:
        category = get_file_name(child_docs_list[0]["source"])

    # TODO : 삭제
    # doc_cur = faq_doc.find({"category": category}).sort({"_id": 1})
    # strt_idx = 1
    # for doc in doc_cur:
    #   print_str = "[{}],[{}],[{}],[{}]".format(doc["category"],doc["page"],strt_idx, doc["content"])
    #   print(print_str)
    #   strt_idx += 1

    return category


"""
  q_list, query_list_txt, inquiry_examples_txt 조회
"""
def get_question_list(faq_qa, category):
    qa_cur = faq_qa.find({"category": category.strip()}).sort({"page": 1})
    strt_idx = 1

    q_list = []
    buff_questions = ""
    buff_examples = ""

    for qa in qa_cur:
        # TODO : 삭제
        print_str = "[{}],[{}],[{}]".format(qa["category"],strt_idx, qa["question"])
        # print(print_str)

        q_list.append(qa["question"])
        buff_questions += "{}.{}\n".format(strt_idx, qa["question"])

        buff_examples += "Inquiry: {}\n".format(qa["question"].split("]")[-1].strip())
        buff_examples += "query:{}\n\n".format(qa["question"])

        strt_idx += 1

    return q_list, buff_questions, buff_examples


"""
  get_question 실행
"""
def get_question(prompt):
    messages = [
        {
            "role": "system",
            "content": "",  # Model not yet trained for follow this
        },
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        # max_new_tokens=128,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        stop_sequence="<|im_end|>",
    )
    print(outputs[0]["generated_text"][-1]["content"])
    # outputs[0]

    question = outputs[0]["generated_text"][-1]["content"]
    if "[" in question:
        question = question[question.find("["):]

    return question


"""
  get_answer_by_question
"""
def get_answer_by_question(faq_qa, question):
    qa_cur = list(faq_qa.find({"question": question}))
    answer = ""
    if len(qa_cur) > 0:
        answer = qa_cur[0]["answer"]

    return answer


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

          You must make a reply very related to Question and Answer sets.
          You should reference Question and Answer sets to generate the reply to Inquiry.
          The reply text content should be within the Answer of Question and Answer sets.
          Do not say that you can not reply.
          Do not refer about 'Question' and 'Answer' sets itself including the No of Question and Answer sets.
          You must always reply in Korean.
          Your answer should be logical and make sense.
          If There are related information (관련 정보), respond it as much as possible.

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
    if t_cnt > 6000:
        query_member_cnt = 2
    if (t_cnt1 + t_cnt2) > 6000:
        query_member_cnt = 1

    return query_member_cnt


"""
  get_answer_by_embedding
"""
def get_answer_by_embedding(embeddings, faq_qa, query):

    query_embedding = embeddings.embed_documents([query.strip()])[0]

    # Retrieve relevant child documents based on query
    child_qas = faq_qa.aggregate([
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding_q",
                "queryVector": query_embedding,
                "numCandidates": 10,
                "limit": 10
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

    child_qas_list = list(child_qas)

    # compute 결과와 child_qas_list 결과를 조합하여 qas list 구성
    child_qas_list, t_cnt1, t_cnt2, t_cnt3 = get_sorted_qas_list(query, child_qas_list)
    print("t_cnt1/t_cnt2/t_cnt3:{}/{}/{}".format(t_cnt1, t_cnt2, t_cnt3))

    default_member_cnt = 3
    query_member_cnt = get_query_member_cnt(default_member_cnt, t_cnt1, t_cnt2, t_cnt3)
    if len(child_qas_list) > query_member_cnt:
        child_qas_list = child_qas_list[:query_member_cnt]
    print("The length of child_qas_list:{}".format(len(child_qas_list)))

    answer = ""
    if len(child_qas_list) > 0:
        print(child_qas_list[0]["question"])
        print("========================================")
        print(child_qas_list[0]["answer"])
        print("========================================")
        print(child_qas_list[0]["score"])

    question = child_qas_list[0]["question"]
    answer = child_qas_list[0]["answer"]
    url = child_qas_list[0]["url"]
    score = float(child_qas_list[0]["score"])

    return question, answer, url, score, child_qas_list


def querying(query, history):

    # faq_doc = get_faq_doc()
    # faq_qa = get_faq_qa()
    nodong_qa = get_nodong_qa()

    # # category
    # category = get_category_by_query(embeddings, faq_doc, query)
    # # q_list, query_list_txt, inquiry_examples_txt
    # q_list, query_list_txt, inquiry_examples_txt = get_question_list(faq_qa, category)
    # # prompt
    # prompt = make_prompt(query, query_list_txt, inquiry_examples_txt)
    # # get_question 실행
    # question = get_question(prompt)

    process_type = "LLM"
    answer = ""
    score = -1
    llm_answer = ""
    # if question in q_list:
    #     print("LLM 성공!")
    #     answer = get_answer_by_question(nodong_qa, question)
    # else:

    process_type = "Embedding"
    question, answer, url, score, qas_list = get_answer_by_embedding(embeddings, nodong_qa, query)

    if score < 0.97:
        llm_answer = get_answer_by_llm(query, qas_list)

    return_text_arr = []
    # return_text_arr.append(f"<h2>Category</h2>\n{category}")
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

query = "취업규칙이 변경되면 기존 근로계약과의 우선순위는 어떻게 달라지게 되나요?"
querying(query, None)
