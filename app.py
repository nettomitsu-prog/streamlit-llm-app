import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# ======================
# 1. .env（APIキー）の読み込み
# ======================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ======================
# 2. LLM関数の定義
# ======================
def get_llm_answer(user_text: str, expert_type: str) -> str:
    """
    入力テキストと専門家タイプを受け取り、
    LLMの回答を文字列で返す
    """

    # 専門家システムメッセージ
    system_prompt = {
        "AIエンジニア": "あなたは優秀なAIエンジニアです。相手に分かりやすく説明してください。",
        "Webマーケター": "あなたはプロのWebマーケターです。売れる文章でアドバイスしてください。"
    }[expert_type]

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.7,
        api_key=OPENAI_API_KEY
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text)
    ]

    result = model.invoke(messages)
    return result.content


# ======================
# 3. Streamlit UI
# ======================
st.set_page_config(page_title="LangChain LLM App", layout="wide")

st.title("🚀 LangChain × OpenAI LLMアプリ")

st.write("""
このWebアプリでは以下のことができます：

- テキストを入力するとAIが回答してくれます  
- **専門家タイプ（AIエンジニア or Webマーケター）**を選べます  
- 選択した専門家視点で回答が変わります  
""")

# ラジオボタン（専門家選択）
expert = st.radio(
    "どのタイプの専門家に相談しますか？",
    ["AIエンジニア", "Webマーケター"]
)

# 入力フォーム
user_input = st.text_area("質問内容を入力してください：", height=120)

# 実行ボタン
if st.button("AIに聞く"):
    if not user_input.strip():
        st.warning("⚠ 入力が必要です")
    else:
        with st.spinner("AIが回答中..."):
            answer = get_llm_answer(user_input, expert)
            st.success("回答はこちら👇")
            st.write(answer)


st.write("---")
st.caption("Powered by LangChain & Streamlit / Python 3.11 でデプロイ可能")
