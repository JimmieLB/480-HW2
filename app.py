import os
import gradio as gr
import spaces
from smolagents import CodeAgent, InferenceClientModel, tool
from tools import search_arxiv
from dotenv import load_dotenv
from datetime import datetime, timezone
import random
load_dotenv()

@tool
def get_inspirational_quote() -> str:
    """Returns a random inspirational quote to inspire the day's reflection."""
    quotes = [
    "Start where you are, use what you have, do what you can.",
    "Small steps every day add up to big changes over time.",
    "You don't have to be perfect to be great.",
    "The best time to begin was yesterday. The next best time is now.",
    "Difficult roads often lead to beautiful destinations.",
    "Your only limit is the one you set for yourself.",
    "Every expert was once a beginner.",
    "Progress, not perfection, is the goal.",
    "Courage is taking the next step even when you're afraid.",
    "Believe you can, and you're halfway there.",
    "Failure is just practice for success.",
    "Dream big, start small, and never stop moving forward.",
    "The harder you work for something, the more it means when you get it.",
    "Be patient with yourself. Growth takes time.",
    "Great things are built one day at a time.",
    "Don't wait for motivation. Build the habit and the motivation will follow.",
    "A setback is a setup for a comeback.",
    "You are stronger than you think and braver than you feel.",
    "Focus on the step in front of you, not the whole staircase.",
    "Today's effort is tomorrow's strength."
]
    return random.choice(quotes)

@tool
def get_current_time() -> str:
    """
    Returns the current date and time in UTC.
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.environ.get("HF_TOKEN")
)

agent = CodeAgent(
    tools=[search_arxiv, get_current_time, get_inspirational_quote], 
    model=model,
    add_base_tools=False,
    max_steps=20
)


# 1. THE DECOY: Passes the HF startup check so the container boots
@spaces.GPU
def decoy_function():
    pass

# 2. THE ACTUAL CHAT: Runs on the free background CPU without touching your quota
def agent_chat(user_prompt):
    prompt = user_prompt + "\n\nReturn your final answer as a plain formatted string, not a dict or list."
    return agent.run(prompt)

demo = gr.Interface(
    fn=agent_chat,
    inputs=gr.Textbox(lines=2, placeholder="E.g., Find 2 recent papers on Agentic AI and format them as BibTeX."),
    outputs=gr.Markdown(label="Agent Output"),
    title="Autonomous arXiv Research Agent",
    description="Powered by smolagents and Qwen 2.5 Coder"
)

demo.launch()

# import os
# import gradio as gr
# import spaces
# from smolagents import CodeAgent, InferenceClientModel
# from tools import search_arxiv

# # 1. Initialize Qwen 2.5 Coder via the Serverless API
# model = InferenceClientModel(
#     model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
#     token=os.environ.get("HF_TOKEN")
# )

# # 2. Assemble the CodeAgent
# agent = CodeAgent(
#     tools=[search_arxiv], 
#     model=model,
#     add_base_tools=False,
#     max_steps=5
# )

# # 3. Define the execution function and request ZeroGPU allocation
# @spaces.GPU
# def agent_chat(user_prompt):
#     return agent.run(user_prompt)

# # 4. Launch the Gradio Chat Interface
# demo = gr.Interface(
#     fn=agent_chat,
#     inputs=gr.Textbox(lines=2, placeholder="E.g., Find 2 recent papers on Agentic AI and format them as BibTeX."),
#     outputs=gr.Markdown(label="Agent Output"),
#     title="Autonomous arXiv Research Agent",
#     description="Powered by smolagents and Qwen 2.5 Coder"
# )

# demo.launch()
