# Scriptscore.AI
This project explores the use of LLMs to predict the success of the TV content (a show or a movie) based on a simple idea. It works by using LLMs to simulate audience sentiment across different audience groups.
The app is built using OpenAI's GPT, Streamlit, langchain, and local persistence with SQLite. 

Further development might include:
* looking into other SOTA models like Claude 3.5 Sonnet.
* exploring debiased models, eg. [Creativity Has Left the Chat: The Price of Debiasing Language Models](https://arxiv.org/abs/2406.05587)

# Demo
![Scriptscore.ai Demo Image](./demo_image.png)

# Run
`streamlit run streamlit_app.py`