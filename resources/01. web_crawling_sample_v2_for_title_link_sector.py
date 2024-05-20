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
nodong_urls_txt = "nodong_urls_hrefs_sector.txt"
urls_arr = get_url_list(nodong_urls_txt)

def url_crawler(url):
    # Send a GET request to the specified URL
    response = requests.get(url)
    a_href_list = []

    print("111111111111111111111")
    print(f"URL : {url}")
    print("222222222222222222222")

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {"class": 'bd_lst'})
        td_list = table.find_all("td", class_="title")
        for td in td_list:
            href = td.find_all("a")[0]["href"]
            a_href_list.append(href)

    return a_href_list

# URL 크롤러 실행
for idx, url in enumerate(urls_arr):
    if not url:
        continue
    # if idx > 0:
    #     break
    a_href_list = url_crawler(url)

    for href in a_href_list:
        print(href)
        file_path = './nodong_urls_sector.txt'
        with open(file_path, 'a', encoding="utf-8") as f:
            f.write("{}\n".format(href))
            f.close()

"""
file_path = './nodong_qa_list.json'
# MongoDB craete
with open(file_path, "r", encoding="utf-8") as f:
    nodong_data_list = json.load(f)
    for d in nodong_data_list:
        print(d["url"])
        print(d["question"])
        print(d["answer"])
"""




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