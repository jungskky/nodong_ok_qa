
import gradio as gr
from util import querying

iface = gr.ChatInterface(
    fn = querying,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="질문을 입력해 주세요.", container=False, scale=7),
    title="노동OK Q&A [(주)유아이네트웍스 AI 챗봇]",
    theme="soft",
    examples=["밤샘근무 다음날 대체휴무로 쉬게 되면 급여는 어떻게 되나요?",
              "실업급여를 부정하게 받으면 어떤 벌을 받게 되나요?",],

    cache_examples=True,
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
    submit_btn="Submit"

    )

iface.launch(share=True)

def on_close():
  iface.set_on_close(on_close)
  iface.launch()
  iface.close()
