from langchain_core.runnables import RunnableLambda

def trivial_echo(input_data: dict) -> dict:
    text = input_data.get("text", "hello") if isinstance(input_data, dict) else str(input_data)
    return {"output": f"Echo: {text}"}

chain = RunnableLambda(trivial_echo)
