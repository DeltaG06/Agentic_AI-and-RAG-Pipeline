from openai import OpenAI
import requests
import json

client = OpenAI ( 
    base_url = "http://localhost:11434/v1",
    api_key="ollama"
    )



def get_weather(city: str ):
    url =f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200 :
        return f"The weather in {city} is {response.text}"
    else :
        return "something went wrong."

available_tools = {
    "get_weather" : get_weather
}



system_prompt =  """

        You're an expert AI in resolving user queries using chain of thought.
        You work on START , PLAN AND OUTPUT steps .
        You need to first Plan what needs to be done .The PLAN can be with multiple steps
        Once you think enough plan has been done , finally you can give an OUTPUT
        You can also call a tool if required from the list of available tools
        for every tool call wait for the observe step which is the output from the called tool
        
        rules:
        -Always respond ONLY with valid JSON.
        -Strictly follow the JSON output format
        -only run on step at a time
        -the sequence of steps is START (where user gives an input ),PLAN(that can be multiple use)
        and finally OUTPUT (that can be displayed to the user).
        -if a block is already executed do not repeat the block and do the same thing twice
        -When using a tool, ONLY use the information returned by the tool.
         Do not invent additional data.

         

        Rules for TOOL step:

        - TOOL must include both "tool" and "input"
        - Never leave "input" empty
        - If multiple cities are requested, call the tool separately for each city
        - After OBSERVE you must produce OUTPUT



        Output JSON format :
        {"step" : "START"||"PLAN"|"OUTPUT"|"TOOL" , "content": "string" ,tool : "string","input":"string"}
        Tool call format:
                {
                "step": "TOOL",
                "tool": "get_weather",
                "input": "city name"
                }
         Final answer:
            {
            "step": "OUTPUT",
            "content": "final answer for the user"
            }

        Available tools :
        -get_Weather(city : str): Takes city names as input and returns the weather info about the city.

        If the user asks weather for multiple cities, return them as a list.

            Example:

            {
            "step": "TOOL",
            "tool": "get_weather",
            "cities": ["goa", "delhi"]
            }

        example 2 :
        START : Hey , WHat is the weather of delhi ?
        PLAN : {"step": "PLAN" , "content": "Seems like the user is interested in getting the weather of delhi"}
        PLAN : {"step":"PLAN" , "content": "Lets see if we have any available tools from the list of available tools"}
        PLAN : {"step":"PLAN" , "content": "Great ! We have get_weather tool avaiable for this query"}
        PLAN : {"step":"PLAN" , "content": "I need to call get weather tool for delhi as inout for city"}
        PLAN : {"step":"TOOL" , tool:"get_weather" , "cities": ["Delhi","goa",etc]}
        PLAN : {"step":"OBSERVE" ,tool:"get_weather" "content": "The temperature is hazy with 29 C"}
        PLAN : {"step":"PLAN" , "content": "Great I got the weather info about delhi"}
        OUTPUT : {"step":"OUTPUT" , "content": "SO the current weather in delhi is 29C with some haze "}


""" 
print("\n\n\n")
message_history = [
    {"role":"system" , "content": system_prompt}
]
user_query = input("->")
message_history.append({"role" : "user" , "content":user_query})

while 1 :
    response = client.chat.completions.create(
        model = "llama3.1:8b",
        response_format = {"type":"json_object"},
        messages = message_history
    )
    raw_result = response.choices[0].message.content
  
    message_history.append({"role":"assistant" , "content": raw_result})
    try:
         parsed_result = json.loads(raw_result)
    except json.JSONDecodeError:
         print("⚠️ Invalid JSON from model:")
         print(raw_result)
         continue


    if parsed_result.get("step") == "START":
        print("🔥" , parsed_result.get("content"))
        continue

    if parsed_result.get("step") == "PLAN":
        print("🧠" , parsed_result.get("content"))
        continue


    if parsed_result.get("step") == "TOOL":

        tool_to_call = parsed_result.get("tool")
        cities = parsed_result.get("cities")

        if tool_to_call not in available_tools:
            print("⚠️ Unknown tool:", tool_to_call)
            continue

        if not cities:
            print("⚠️ No cities provided")
            continue

        results = []

        for city in cities:
            tool_response = available_tools[tool_to_call](city)
            print(f"✂️ {tool_to_call} {city} ==> {tool_response}")

            results.append({
                "city": city,
                "weather": tool_response
            })

        message_history.append({
            "role": "developer",
            "content": json.dumps({
                "step": "OBSERVE",
                "tool": tool_to_call,
                "results": results
            })
        })

        continue
    


    if parsed_result.get("step") == "OUTPUT":
        print("🎂", parsed_result.get("content"))
        print("DEBUG:", parsed_result)
        break
        
print("\n\n\n")