# 本地知识库大模型 langchain + LLaMa + Custom Agent

本例主要参考 [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM) 和 [langchain-GLM_Agent](https://github.com/jayli/langchain-GLM_Agent) 。在此基础上，部分模块进行了自定义替换，以搭建一个基于本地知识库、在线检索、本地LLM服务的问答系统，


## 准备工作

1. 本例采用 13B 的 LLama-base 的模型，用户可根据需要在 `models/custom_llm.py` 中替换
1. 使用在线搜索需要在 `models/custom_search.py` 中设置你的 `RapidAPIKey = ""`，可在[这里](https://rapidapi.com/microsoft-azure-org-microsoft-cognitive-services/api/bing-web-search1)申请

代码checkout下来后，执行

```
python cli.py
```

即可实现全过程



