import openai
import phi.api
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import phi
from phi.playground import Playground, serve_playground_app
import os 

load_dotenv()

phi.api = os.getenv("PHI_API_KEY")


# web search agent 
web_search_agent= Agent(

    name = "web_search_agent",
    role= "search the web for the information",
    model = Groq( id ="llama-3.3-70b-versatile"),
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


app = Playground(agents = [web_search_agent, financial_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)