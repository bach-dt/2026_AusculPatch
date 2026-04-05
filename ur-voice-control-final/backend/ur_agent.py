from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import system_prompt
from typing import List

from tools import ur_tools, calculation, camera_tool, drawing_tool

import os
import inspect

class URAgent: 
  def __init__(self, streaming: bool = True, verbose: bool = True, save_history: bool = True,  model: str = "ChatGPT"):
    self.__streaming = streaming
    self.__verbose = verbose
    self.__model = model
    self.__save_history = save_history
    self.__chat_history = []

    self.__llm = self.get_llm(self.__model, self.__streaming)
    self.__tools = self.get_tools([ur_tools, calculation, camera_tool, drawing_tool])
    self.__prompts = self.get_prompts()

    self.__agent = self.get_agent()
    self.__executor = self.get_executor(self.__verbose)

  def invoke(self, message: str) -> str: 
    result = self.__executor.invoke(
      {"input": message, "chat_history": self.__chat_history}
    )
    self.record_history(message, result["output"])
    return result["output"]

  def get_agent(self):
    return create_tool_calling_agent(
      llm=self.__llm,
      tools=self.__tools,
      prompt=self.__prompts
    )

  def get_executor(self, verbose: bool) -> AgentExecutor:
    return AgentExecutor(
      agent=self.__agent,
      tools=self.__tools,
      verbose=verbose
    )

  def get_prompts(self):
    template = ChatPromptTemplate.from_messages(
      system_prompt
      + [
          MessagesPlaceholder(variable_name="chat_history"),
          ("user", "{input}"),
          MessagesPlaceholder(variable_name="agent_scratchpad"),
      ]
    )
    return template
  
  def get_llm(self, model: str, streaming: bool):
    if model == "ChatGPT":
      openai_api_key = self.get_env_variable("OPENAI_API_KEY")

      llm = ChatOpenAI(
        model_name="gpt-4.1",  
        openai_api_key=openai_api_key,
        temperature=0,  
        streaming=streaming,
      )

      return llm
    # Add Gemini Later
  
  def get_env_variable(self, name: str) -> str: 
    value = os.getenv(name)
    if value is None:
      raise ValueError(f"Environment variable {name} is not set.")
    return value

  def get_tools(self, packages: List):
    tools = []
    for package in packages: 
      for name, obj in inspect.getmembers(package):
        if isinstance(obj, BaseTool):
          tools.append(obj)

    return tools

  def record_history(self, message: str, response: str):
    if self.__save_history: 
      self.__chat_history.extend(
        [HumanMessage(content=message), AIMessage(content=response)]
      )

  def set_llm(self, model):
    pass