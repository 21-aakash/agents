from phi.agent import Agent #Base class for creating agents.
from phi.model.groq import Groq #Class for creating and managing Groq models.
from phi.tools.yfinance import YFinanceTools #Tools for interacting with Yahoo Finance data.
from phi.tools.duckduckgo import DuckDuckGo #Tools for performing searches using DuckDuckGo.
import openai
import os
from dotenv import load_dotenv


# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()

# Set the Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")




# web search agent 
web_search_agent= Agent(

    name = "web_search_agent",
    role= "search the web for the information",
    model = Groq( id ="llama-3.3-70b-versatile" , api_key= GROQ_API_KEY),
    tools = [DuckDuckGo()],
    instructions= ["always include the sources"],
    show_tools_calls= True,
    markdown= True
)


# financial agent

financial_agent= Agent(

    name = "financial AI agent",
     tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, 
                          company_news = True)],
        model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),

    show_tool_calls=True,
    instructions=["use tables to display the data"],
    markdown=True,

)


multi_ai_agent= Agent(
    team =[web_search_agent, financial_agent],
      model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include the sources" , "use table to display the data "],
    markdown=True,
    show_tool_calls=True,

)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDIA" , Steam = True)