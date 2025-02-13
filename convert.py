#!/usr/bin/env python
# coding: utf-8

# v5
# 
# 引用元を保持するエージェント
# 
# googleドキュメントに出力する
# 

# In[2]:


ver = "ver5"


# ライブラリのインポート

# In[3]:


# ライブラリのインポート
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from typing import List, Dict
from langchain.schema import SystemMessage, HumanMessage
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

load_dotenv()


#LangchainのAPIを渡す
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

#GPTのAPIを渡す
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#TavilyのAPIを渡す
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


# 出力の型を定義

# In[4]:


class URLInfo(BaseModel):
    title: str = Field(description="サイトのタイトル")
    url: str = Field(description="サイトのURL")

class OutputSummrize(BaseModel):
    # URL_listをURLInfoモデルに変更
    URL_list: List[URLInfo] = Field(description="記事作成に利用されたサイトのタイトルとURL")


class OutputArticle(BaseModel):
    title: str = Field(description="記事のタイトル")
    main_text: str = Field(description="記事の中身")


# プロンプトの定義

# In[5]:


search_system_template = '''
あなたはユーザーから与えられたトピックについて調査を行うアシスタントです。
調査で得られた情報と引用したサイトのタイトルとURLを出力してください。
'''

summrize_system_template = '''
あなたは調査で得られたサイトの情報を集約するアシスタントです。
調査で取得したサイトのタイトルとURL１つ１つを辞書形式({{"title": "...", "url": "..."}})にし、リスト形式で出力してください。
'''

generate_system_template = '''
以下の内容を踏まえて、マークダウン形式で記事を作成してください。

指定されたコンテンツ {content} に関する記事を日本語で作成してください。
検索結果 {search_result} から得られる情報を整理し、読者にとって有益でわかりやすい記事を心がけてください。
記事のタイトル、導入、本文、まとめの構成を意識し、最新かつ正確な情報を盛り込んでください。
本文の末尾には、検索結果のサイトタイトルとURLを{URL_list}を参照の上、リスト形式で記載してください。

Steps
検索結果の内容を把握
search_resultを確認し、主要なトピックやキーワードを整理してください。
記事のアウトライン作成
タイトル
導入（リード文）
見出し（必要に応じて複数）
まとめ（結論やまとめ）
本文執筆
見出しや小見出しごとに論点をまとめ、論理的に構成してください。
検索結果から得られた情報や一般知識を補完し、正確かつオリジナリティを意識した内容にしてください。
結論・まとめ
記事全体を振り返り、要点を簡潔にまとめてください。
次の行動や関連トピックへの誘導など、読者が得られるメリットを提示してください。
最終チェック
誤字脱字、情報の正確性、読みやすさを確認し、必要に応じて修正してください。
引用元の表記
記事の末尾に、検索結果から参照したサイトのタイトルとURLをリスト形式で記載してください。
Output Format
マークダウン形式で出力してください。
改行や段落を保持し、見出しや箇条書きを適切に使ってください。
記事末尾に「引用元」として、サイトタイトルとURLを記載してください。
Examples
例:

タイトル: 「[検索結果の主要トピック]を徹底解説」
導入: この記事では…（簡単な背景情報と読者が得られるメリット）

本文

見出し1: 「[見出しのテーマ]」
[検索結果からの情報を例示しながら解説]
見出し2: 「[別の角度や詳細トピック]」
[具体的な事例や追加情報など]
まとめ: [全体の要点、読者への次のステップ]

引用元

サイト名
サイト名
(実際の記事では、上記より詳細かつ具体的に説明してください)

Notes
search_result や contentが十分でない場合は、想定しうる前提条件を補足して文章を構成してください。
あくまで検索結果を根拠にしつつも、独自の言い回しや整理を行い、オリジナリティを大切にしてください。
'''



# エージェントの定義

# In[6]:


#ツールを利用するエージェント
#ChatOpenAIでチャットモデルを渡す（モデルは変更できる）
model = ChatOpenAI(model="gpt-4o-mini")

# Tavilyの定義　max_results:何件文の検索結果を保持するか
search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True
)
tools = [search]

search_agent = create_react_agent(model, tools)


# In[7]:


#URLを保持するエージェント
summrize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", summrize_system_template)
    ]
)

summrize_output = model.with_structured_output(
    OutputSummrize
)

summrize_agent = summrize_prompt | summrize_output


# In[8]:


#記事作成を行うエージェント
extract_prompt = ChatPromptTemplate.from_messages(
    [
        ("user", generate_system_template),
    ]
)

extract_trends = model.with_structured_output(
    OutputArticle
)
trend_extractor = extract_prompt | extract_trends


# stateの定義

# In[9]:


class State(TypedDict):
    content: str #ユーザーが入力した記事の内容
    search_result: str #LLMが調査を行った結果を格納
    title: str #生成された記事のタイトル
    main_text:str #生成された記事の内容
    URL_list:List[Dict[str, str]] #URLのリスト


# 各ノードの関数

# In[10]:


def search(state: State):
    response = search_agent.invoke(
        {"messages": [
            {"role": "system", "content": search_system_template},
            {"role": "user", "content": state["content"]}
            ]
        }
    )
    state["search_result"] = response["messages"][-1].content
    return state

def search_summrize(state: State):
    response = summrize_agent.invoke(state)
    state["URL_list"] = response.URL_list
    return state

def generate_article(state: State):
    response = trend_extractor.invoke(state)
    state["title"] = response.title
    state["main_text"] = response.main_text
    return state


# グラフの作成

# In[11]:


#グラフの定義
workflow = StateGraph(State)

#ノードの定義
workflow.add_node("search", search)
workflow.add_node("search_summrize", search_summrize)
workflow.add_node("generate_article", generate_article)

#エッジの定義
workflow.add_edge(START, "search")
workflow.add_edge("search", "search_summrize")
workflow.add_edge("search_summrize", "generate_article")
workflow.add_edge("generate_article", END)

#コンパイル
app = workflow.compile()


# In[12]:


#グラフの可視化
from IPython.display import Image, display

display(Image(app.get_graph(xray=True).draw_mermaid_png()))
from IPython.display import Image, display


# トピックの定義

# In[13]:


initial_state = State()
initial_state["content"] = "大谷翔平"


# グラフの実行

# In[14]:


output = app.invoke(initial_state)
output


# ドキュメント化

# In[15]:


markdown_text = f"""
{output['title']}

{output['main_text']}

"""
file_name = f"{ver}_{output['title']}.md"
folder = "記事格納"
file_path = os.path.join(folder, file_name)

# Markdownファイルとして保存する（.mdの拡張子推奨）
with open(file_path, "w", encoding="utf-8") as file:
    file.write(markdown_text)

print(f"{file_name} に保存しました。")


# ストリーミングでの取得方法

# In[16]:


'''
for event in app.stream(initial_state):
    for k, v in event.items():
        if k != "__end__":
            print(v)
'''  

