from crewai import Agent, Task, Crew , LLM
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

Game_Designer = Agent(
    role="specialist game designer",
    goal = "Make a game with unique features and engaging gameplay mechanics.",
    backstory = "an expert game designer with 10 years of experience in designing popular games.",
    llm=llm,
)
design_task = Task(
    description="Create a comprehensive game design document for a game similar to Minecraft, including unique features and engaging gameplay mechanics.",
    expected_output="a detailed game design document outlining the game's concept, mechanics, features, and target audience.",
    agent=Game_Designer,
)
Game_Reviewer = Agent(
    role="game design reviewer",
    goal="Review the game design document created by the game designer for creativity, engagement, and feasibility.",    
    backstory="an experienced game reviewer with a keen eye for detail and a passion for innovative game design.",
    llm=llm,
)

review_task = Task(
    description="Review the game design document created by the game designer for creativity, engagement, and feasibility.",
    expected_output="a detailed review highlighting strengths and areas for improvement also rate the game from 1 to 10.",
    agent=Game_Reviewer,
)

Game_Improver = Agent(
    role="game design improver",   
    goal="Improve the game design document based on the review provided by the game reviewer.",
    backstory="a skilled game designer with expertise in refining and enhancing game concepts.",   
    llm=llm,
)
improve_task = Task(
    description="Improve the game design document based on the review provided by the game reviewer, you will improve the document provided by the game designer.",
    expected_output="an enhanced game design document that addresses the feedback and suggestions from the reviewer.",
    agent=Game_Improver,
)

crew = Crew(
    agents=[Game_Designer, Game_Reviewer, Game_Improver],
    tasks=[design_task, review_task, improve_task],
    verbose=True,
    llm=llm
)
result = crew.kickoff()
print(result)


