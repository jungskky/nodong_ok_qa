import requests
import json
from bs4 import BeautifulSoup

urls_arr = []
# urls_arr.append("https://www.nodong.kr/index.php?mid=lowpay&category=413191&document_srl=402953")
# urls_arr.append("https://www.nodong.kr/severance_pay/2278954")
# urls_arr.append("https://www.nodong.kr/lowpay/851345")
# urls_arr.append("https://www.nodong.kr/gabjil/2008899")
# urls_arr.append("https://www.nodong.kr/harassment/403824")
# urls_arr.append("https://www.nodong.kr/holyday/403433#0")
# urls_arr.append("https://www.nodong.kr/severance_pay/403403")
# urls_arr.append("https://www.nodong.kr/imgum/1722194#0")
# urls_arr.append("https://www.nodong.kr/interpretation/2383551")
# urls_arr.append("https://www.nodong.kr/imgum/402802#0")
# urls_arr.append("https://www.nodong.kr/pds/403567")
# urls_arr.append("https://www.nodong.kr/index.php?mid=restructuring&category=456990&document_srl=406849")
# urls_arr.append("https://www.nodong.kr/restructuring/406848")
# urls_arr.append("https://www.nodong.kr/restructuring/406855")#
# urls_arr.append("https://www.nodong.kr/restructuring/406838")#
# urls_arr.append("https://www.nodong.kr/restructuring/406835")
# urls_arr.append("https://www.nodong.kr/restructuring/406827")
# urls_arr.append("https://www.nodong.kr/index.php?mid=restructuring&category=456990&document_srl=406847")
# urls_arr.append("https://www.nodong.kr/index.php?mid=restructuring&category=456986&document_srl=406855")
# urls_arr.append("https://www.nodong.kr/index.php?mid=restructuring&category=456992&document_srl=406844")
# urls_arr.append("https://www.nodong.kr/index.php?mid=restructuring&category=457000")




# 중복제거
def remove_overlab(a_arr):
    a_arr_check_list = []
    for a in a_arr:
        if a.text:
            if a not in a_arr_check_list:
                a_arr_check_list.append(a)
    return a_arr_check_list

def get_url_list(nodong_urls_txt):
    # URL 텍스트 파일 Open
    with open(nodong_urls_txt, "r", encoding="utf8") as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() in urls_arr:
                continue
            # TODO : 삭제
            print(line.strip())
            urls_arr.append(line.strip())
    return urls_arr

# url 목록
nodong_urls_txt = "nodong_urls_best_qa.txt"
urls_arr = get_url_list(nodong_urls_txt)

def url_crawler(url):
    # Send a GET request to the specified URL
    response = requests.get(url)
    h1_text = ""
    text_buff = ""

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        # soup.find("hr").decompose()

        # ------------------------
        div_top = soup.find('div', {"class": 'top_area'})

        # h1_text = ""
        # if div_top:
        #     div_top_h1 = div_top.findChild("h1")
        #     h1_text = div_top_h1.text

        # top_a = div_top.findChild("a")
        # top_a_text = div_top.findChild("a").text


        h1_text = div_top.findChild("a").text

        div_contents = soup.find('div', {"class": 'xe_content'})

        # TODO : 테스트 후 삭제
        div_contents_children = div_contents.findChildren(recursive=False)
        div_contents_children_02 = div_contents.findChildren()

        # TODO : 테스트 후 삭제
        h_arr = []
        c_arr_tmp =[]
        # ------------------------------------------
        div_contents_children = div_contents.findChildren(recursive=False)

        # 중복제거 링크 목록
        a_arr = remove_overlab(div_contents.findChildren("a"))

        for idx, c in enumerate(div_contents_children):

            # TODO : 테스트 후 삭제
            if c.name in ("hr", "h1", "h2", "h3", "h4") and len(c_arr_tmp) > 0:
                # print("{}".format(c.text))
                h_arr.append(c_arr_tmp)
                c_arr_tmp = []

            c_arr_tmp.append(c)

        if len(c_arr_tmp) > 0:
            h_arr.append(c_arr_tmp)
            c_arr_tmp = []

        # TODO : H1 print
        print(h1_text)

        if len(h_arr) > 0:
            for idx, c1 in enumerate(h_arr):

                for c2 in c1:
                    text_tmp = c2.text

                    # ###############################
                    # a link 추가
                    # ###############################
                    for a in a_arr:
                        if a.text in c2.text:
                            text_tmp = str(text_tmp).replace(a.text, str(a))
                    
                    # TODO : 삭제
                    print("{}".format(text_tmp))

                    text_tmp = text_tmp.replace("답변", "답변:")
                    text_tmp = text_tmp.replace("관련 정보", "관련 정보:")
                    text_tmp = text_tmp.replace("더 자세한 정보", "더 자세한 정보:")
                    text_buff += "\n{}\n".format(text_tmp)

    return url, h1_text, text_buff

json_member_list = []

# URL 크롤러 실행
for idx, url in enumerate(urls_arr):
    if not url:
        continue
    # if idx > 0:
    #     break
    url, h1_text, text_buff = url_crawler(url)
    print("url:{}".format(url))
    print("h1_text:{}".format(h1_text))
    print("text_buff:{}".format(text_buff))

    json_member_list.append({"url":url, "question":h1_text, "answer":text_buff})


file_path = './nodong_qa_list_best_qa.json'
with open(file_path, 'w', encoding="utf-8") as f:
    json.dump(json_member_list, f, indent=4, ensure_ascii=False)
    f.close()




"""
encode_contents
replace_with
unwrap
find_all
find_all_next
find_all_previous
find_next
find_next_sibling
find_next_siblings
find_parent
find_parents
find_previous
find_previous_sibling
find_previous_siblings
next_sibling
previous_sibling
"""