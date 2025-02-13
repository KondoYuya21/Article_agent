#!/usr/bin/env python
# coding: utf-8

# v1 

# ライブラリのインポート

# In[32]:


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
from typing import List
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

# In[33]:


class OutputArticle(BaseModel):
    title: str = Field(description="記事のタイトル")
    main_text: str = Field(description="記事の中身")


# LLMの定義

# In[34]:


#ツールを利用するエージェント
#ChatOpenAIでチャットモデルを渡す（モデルは変更できる）
model = ChatOpenAI(model="gpt-4o-mini")

# Tavilyの定義　max_results:何件文の検索結果を保持するか
search = TavilySearchResults(
    max_results=2,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True
)
tools = [search]

agent_executor = create_react_agent(model, tools)


# In[35]:


#記事作成を行うエージェント
system_template = '''
あなたは記事作成を行うエージェントです。これから与えられる##検索結果を用いて、##コンテンツについての記事を日本語で作成してください

##検索結果
{search_result}

##コンテンツ
{content}

'''

extract_prompt = ChatPromptTemplate.from_messages(
    [
        ("user", system_template),
    ]
)
extract_trends = model.with_structured_output(
    OutputArticle
)
trend_extractor = extract_prompt | extract_trends


# グラフの作成

# In[36]:


class State(TypedDict):
    content: str #ユーザーが入力した記事の内容
    search_result: str #LLMが調査を行った結果を格納
    title: str #生成された記事のタイトル
    main_text:str #生成された記事の内容


# In[37]:


def search_agent(state: State):
    response = agent_executor.invoke({"messages": [{"role": "user", "content": f"{state['content']}について調査をして、まとめてください"}]})
    state["search_result"] = response["messages"][-1].content
    return state

def generate_article(state: State):
    response = trend_extractor.invoke(state)
    state["title"] = response.title
    state["main_text"] = response.main_text
    return state


# In[38]:


workflow = StateGraph(State)

workflow.add_node("search_agent", search_agent)
workflow.add_node("generate_article", generate_article)

workflow.add_edge(START, "search_agent")
workflow.add_edge("search_agent", "generate_article")
workflow.add_edge("generate_article", END)

app = workflow.compile()


# In[39]:


from IPython.display import Image, display

display(Image(app.get_graph(xray=True).draw_mermaid_png()))
from IPython.display import Image, display


# In[42]:


initial_state = State()
initial_state["content"] = "大谷翔平"


# In[43]:


for event in app.stream(initial_state):
    for k, v in event.items():
        if k != "__end__":
            print(v)


# In[ ]:




