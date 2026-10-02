from langchain_core.messages import HumanMessage
from src.agent import get_workflow



if __name__ == "__main__":
    # source_code_docs = [
    #         "To authenticate API requests, use the Client(api_key) class constructor.",
    #         "The function database.connect() requires a URI string and establishes a pool of 5 workers."
    #     ]
    config = {"configurable": {"thread_id": "user_session_1"}}
    app = get_workflow()

    query = input("Hi, I'm Priscilla! What can I help you with?\n")

    while query.lower() != "exit":
        for event in app.stream({"messages": [{"role": "user", "content": query}]}, config):
            for value in event.values():  
                print(value["messages"][-1].content)
        query = input()