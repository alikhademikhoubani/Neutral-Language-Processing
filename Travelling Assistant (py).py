from typing import TypedDict, Annotated, List, Literal, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import PyPDF2 as pdf
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import requests

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model = "llama-3.3-70b-versatile")

#origin = input("\nPlease enter IATA code of origin city: ")

#origin2 = input("\nPlease enter the name of origin city: ")

#destination = input("\nPlease enter IATA code of destination city: ")

#destination2 = input("\nPlease enter the name of destination city: ")

#date_of_travel = input("\nPlease enter data of travel like this: 2025-11-12: ")

input_text = input("\nYou: ")

prompt1 = f"just give me one of the IATA name of origin city from this text. just give me the IATA code: {input_text}"
origin_resp = llm.invoke(prompt1)
origin = origin_resp.content
print(f"IATA code of origin city: {origin}")

prompt2 = f"just extract the name of origin city from this text: {input_text}"
origin2_resp = llm.invoke(prompt2)
origin2 = origin2_resp.content
print(f"Name of origin city: {origin2}")


prompt3 = f"just give me one of the IATA name of destination city from this text. just give me the IATA code: {input_text}"
des_resp = llm.invoke(prompt3)
destination = des_resp.content
print(f"IATA code of destination city: {destination}")

prompt4 = f"just extract the name of destination city from this text: {input_text}"
des2_resp = llm.invoke(prompt4)
destination2 = des2_resp.content
print(f"name of destination city: {destination2}")

prompt5 = f"just extract the date in this text: {input_text}"
date_resp = llm.invoke(prompt5)
date_of_travel = date_resp.content
print(f"date of travel: {date_of_travel}")



class SupervisorState(MessagesState):
    """State for the multi-agent"""
    next_agent: str = "supervisor"
    flight_data: str = ""
    weather_data: str = ""
    final_report: str = ""
    task_complete: bool = False
    current_task: str = ""



def create_supervisor_chain():
    """Creates the supervisor decision chain"""
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor managing a team of agents:

1. flight - Gathers information about flights between two cities
2. weather - Gathers information about weather of the city
3. Writer - Creates reports and summaries

Based on the current state and conversation, decide which agent should work next.
if the task is complete, respond with 'DONE'.

Current state:
- Has flight data: {has_flight}
- Has weather data: {has_weather}
- Has report: {has_report}

Respond with ONLY the agent name (flight/weather/writer) or 'DONE'.
"""),
        ("human", "{task}")
    ])

    return supervisor_prompt | llm



def supervisor_agent(state: SupervisorState) -> Dict:
    messages = state.get("messages", [])
    current_task = state.get("current_task", "")

    if not current_task:
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        if human_msgs:
            current_task = human_msgs[-1].content.strip()
        else:
            current_task = "No task"

    has_flight = bool(state.get("flight_data"))
    has_weather = bool(state.get("weather_data"))
    has_report = bool(state.get("final_report"))

    chain = create_supervisor_chain()
    decision = chain.invoke({
        "task": current_task,
        "has_flight": has_flight,
        "has_weather": has_weather,
        "has_report": has_report
    })

    decision_text = decision.content.strip().lower()
    print(f"Supervisor decision: {decision_text}")

    if "done" in decision_text or has_report:
        next_agent = "__end__"
        supervisor_msg = "Supervisor: All tasks complete! Great work team."
    elif "weather" in decision_text or (has_flight and not has_weather):
        next_agent = "weather"
        supervisor_msg = "Supervisor: flight search done. Time for weather search. Assigning to Analyst..."
    elif "writer" in decision_text or (has_weather and not has_report):
        next_agent = "writer"
        supervisor_msg = "Supervisor: weather searching is completed. Let's create the report. Assigning to Writer..."
    else:
        next_agent = "flight"
        supervisor_msg = "Supervisor: Starting with flight. Assigning to flight..."

    return {
        "messages": state["messages"] + [AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": current_task  
    }



def flight_agent(state: SupervisorState) -> Dict:
    """Gathers information about flights"""

    api_key = "lNXHKtMRYtRdUiBX6gOnC6IANNmSjdr2"
    api_secret = "0c4xlMMO9wsBpZOm" 

    url = "https://test.api.amadeus.com/v1/security/oauth2/token"

    data = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": api_secret
    }

    response = requests.post(url, data=data)


    access_token = response.json().get("access_token")
 

    origin_city_code = origin 
    destination_city_code = destination
    departure_date = date_of_travel

    url = f"https://test.api.amadeus.com/v2/shopping/flight-offers"

    params = {
        "originLocationCode": origin_city_code,
        "destinationLocationCode": destination_city_code,
        "departureDate": departure_date,
        "adults": 1, 
        "max": 2 
    }

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(url, headers=headers, params=params)

    flights_info = ""

    data = response.json()
    flights = data.get("data", [])
        
    for flight in flights:
        flight_id = flight.get("id")
        price = flight.get("price", {}).get("grandTotal")
        duration = flight.get("itineraries", [{}])[0].get("duration")
            
        flights_info += f"Flight ID: {flight_id}\n"
        flights_info += f"Price: {price} EUR\n"
        flights_info += f"Duration: {duration}\n"
        flights_info += "-" * 24 + "\n"
    
    return {
        "messages": state["messages"] + [AIMessage(content = flights_info)],
        "flight_data": flights_info,
        "next_agent": "supervisor"
    }



def weather_agent(state: SupervisorState) -> Dict:
    """Gathers information about weather"""
    api_key = "68039a20685c0c855c08ed1f136a33bf" 
    city = destination2
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric&lang=fa"

    response = requests.get(url)
    data = response.json()

    weather_info = ""

    if response.status_code == 200:
        weather_info += f"City: {city}\n"
        weather_info += f"Temperature: {data['main']['temp']}°C\n"
        weather_info += f"Humidity: {data['main']['humidity']}%\n"
        weather_info += f"Weather: {data['weather'][0]['description']}\n"
    else:
        weather_info = "Error fetching data"
    
    return {
        "messages": state["messages"] + [AIMessage(content = weather_info)],
        "weather_data": weather_info,
        "next_agent": "supervisor"
    }



def writer_agent(state: SupervisorState) -> Dict:
    """Writer uses Groq to create final report"""
    flight_data = state.get("flight_data", "")
    weather_data = state.get("weather_data", "")
    task = state.get("current_task")

    writing_prompt = f"""
You are an expert report writer. Your job is to prepare a report on the cost of airfare and the weather of the destination for a person who is planning to travel, based on the specified task and the flight information and weather information provided, so that the person can find complete information about these matters.

task: {task}

flight data: {flight_data}

weather data: {weather_data}

The length of this report should be 1000 words. 
To separate each section of the report, use separators such as "---". 
Please write a comprehensive report of both flight data and weather data based on the specified task.
"""
    
    report_response = llm.invoke([HumanMessage(content = writing_prompt)])
    report = report_response.content

    final_report = f"""
    final report: 
{'=' * 50}
task: {task}
{'=' * 50}

{report}
"""
    
    return {
        "messages": state["messages"] + [AIMessage(content = "Analyst: I've completed the analysis.\n\nTop insights:\n{report}...")],
        "final_report": final_report,
        "next_agent": "supervisor",
        "task_complete": True
    }



def router(state: SupervisorState):
    next_agent = state.get("next_agent", "supervisor")
    
    if next_agent in ["__end__", "end"] or state.get("task_complete", False):
        return END
    
    if next_agent in ["supervisor", "flight", "weather", "writer"]:
        return next_agent
    
    return "supervisor"



workflow = StateGraph(SupervisorState)

workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("flight", flight_agent)
workflow.add_node("weather", weather_agent)
workflow.add_node("writer", writer_agent)

workflow.set_entry_point("supervisor")

for node in ["supervisor", "flight", "weather", "writer"]:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "supervisor": "supervisor",
            "flight": "flight",
            "weather": "weather",
            "writer": "writer",
            "__end__": END
        }
    )

graph = workflow.compile()



#response = graph.invoke({
#    "messages": [HumanMessage(content=f"I want to fly from {origin2} to {destination2}. How much does it cost to fly and what is the weather like there right now?")]
#})

response = graph.invoke({"messages": [HumanMessage(content = input_text)]})


flight_output = response['messages'][2].content
weather_output = response['messages'][4].content
writer_output = response['messages'][6].content
final_report = response['final_report']

print("Flight agent Output:\n", flight_output)
print("Weather agent Output:\n", weather_output)
print("Writer Output:\n", writer_output)
print(final_report)