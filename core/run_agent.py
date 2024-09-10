from langchain_core.messages.ai import AIMessage
from core.state_graph import customer_support  # Import the compiled StateGraph
from core.helpers import _print_event


def get_full_response(test_questions: list[str], passenger_id: str = "0", thread_id: str = "0", debug: bool = False) -> list[str]:
    config = {
        "configurable": {
            "passenger_id": passenger_id,
            "thread_id": thread_id,
        }
    }

    _printed = set()
    responses = []
    
    for question in test_questions:
        events = customer_support.stream( 
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        for event in events:
            responses.append(event)
            if debug:
                _print_event(event, _printed)

    return responses



def get_ai_response(user_input: str, passenger_id: str = "0", thread_id: str = "0", debug: bool = False) -> str:
    config = {
        "configurable": {
            "passenger_id": passenger_id,
            "thread_id": thread_id,
        }
    }

    response = None
    for event in customer_support.stream({"messages": ("user", user_input)}, config):
        for value in event.values():
            if debug:
                print("Assistant:", value["messages"].content)
            try:
                response = value["messages"].content
            except AttributeError:
                response = value

    return response


def get_user_responses(test_questions: list[str], passenger_id: str = "0", thread_id: str = "0", debug: bool = False) -> list[str]:
    config = {
        "configurable": {
            "passenger_id": passenger_id,
            "thread_id": thread_id,
        }
    }

    _printed = set()
    user_responses = []
    
    for question in test_questions:
        events = customer_support.stream( 
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        final_response = ""
        for event in events:
            if "messages" in event:
                for message in event["messages"]:
                    if isinstance(message, AIMessage):
                        final_response = message.content
            if debug:
                _print_event(event, _printed)
        if final_response:
            user_responses.append(final_response)

    return user_responses
