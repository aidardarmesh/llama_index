from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool
from llama_parse import LlamaParse

# settings
Settings.llm = OpenAI(model="gpt-3.5-turbo",temperature=0)

# function tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

documents2 = LlamaParse(result_type="markdown").load_data(
    "./data/2023_canadian_budget.pdf"
)
index2 = VectorStoreIndex.from_documents(documents2)
query_engine2 = index2.as_query_engine()

# response = query_engine.query("What was the total amount of the 2023 Canadian federal budget?")
# print(response)

# rag pipeline as a tool
budget_tool = QueryEngineTool.from_defaults(
    query_engine2,
    name="canadian_budget_2023",
    description="A RAG engine with some basic facts about the 2023 Canadian federal budget."
)

agent = ReActAgent.from_tools([multiply_tool, add_tool, budget_tool], verbose=True)

response = agent.chat(
    "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?"
)

print(response)

response = agent.chat(
    "How much was allocated to a implement a means-tested dental care program in the 2023 Canadian federal budget?"
)

print(response)

response = agent.chat(
    "How much was the total of those two allocations added together? Use a tool to answer any questions."
)

print(response)
