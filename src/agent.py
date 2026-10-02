from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field
from src.retrieval import get_retriever

# Define the shared Agent State
class AgentState(TypedDict):
    # `list` specifies that messages is gonna be a list and add_messages will be used for adding to the list of messages
    messages: Annotated[list, add_messages]


# Define Agent Node
def create_agent_node(llm):
    def agent(state: AgentState):
        # For a node, take several messages and return new messages that are a list. llm.invoke adds messages to the state
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    return agent


def get_workflow():
    # Initialize local LLM with tool binding capability
    llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

    
    # retriever = get_retriever(source_code_docs)

    # retriever_tool = create_retriever_tool(
    #     retriever, 
    #     name="code_base_search", 
    #     description="Reads a codebase to write and evaluate code"
    # )
    #tools = [retriever_tool]
    #llm_with_tools = llm.bind_tools(tools)


    # Build the Graph
    workflow = StateGraph(AgentState)

    # Add processing nodes
    workflow.add_node("agent", create_agent_node(llm))

    # Establish connections and conditional loops
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    #
    memory = MemorySaver()
    # Compile graph into an executable application
    app = workflow.compile(checkpointer=memory)
    return app