from openai import OpenAI
from pydantic import BaseModel, Field
import requests
import json
from typing import Optional, List

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text.strip()}"
        return f"Could not fetch weather for {city} (status {response.status_code})"
    except requests.RequestException as e:
        return f"Network error for {city}: {str(e)}"

available_tools = {
    "get_weather": get_weather
}


system_prompt = """
You are a weather assistant. Respond ONLY in valid JSON. No text outside JSON ever.

You have exactly THREE valid steps: THINK, TOOL, OUTPUT.

## STEP 1 — THINK
Reason about what the user wants and what you need to do.
{"step": "THINK", "content": "your reasoning here"}

## STEP 2 — TOOL
Call this to fetch weather data.
{"step": "TOOL", "tool": "get_weather", "cities": ["city1", "city2"]}

## STEP 3 — OUTPUT
Call this ONLY after you receive weather results from the system.
{"step": "OUTPUT", "content": "your answer here"}

## STRICT RULES
- Your FIRST response must ALWAYS be a THINK step.
- Your SECOND response must ALWAYS be a TOOL step.
- After the system provides weather results, your NEXT response must ALWAYS be OUTPUT.
- In THINK, explicitly reason about: what cities are needed, what tool to use, what to watch out for.
- NEVER output a step called OBSERVE. That is a system-only label.
- NEVER guess or invent weather. Copy exact values from system results.
- Only three valid steps: THINK, TOOL, OUTPUT. Nothing else.
- When you

## EXAMPLE

User: "weather in Delhi and Mumbai?"

You respond:
{"step": "THINK", "content": "The user wants weather for two cities: Delhi and Mumbai. 
I need to call get_weather with both cities in the list. 
After I receive the results I must report the exact temperature and condition 
without modifying or guessing any values."}

You respond:
{"step": "TOOL", "tool": "get_weather", "cities": ["Delhi", "Mumbai"]}

System injects results:
{"results": [
  {"city": "Delhi", "weather": "The weather in Delhi is Haze +38°C"},
  {"city": "Mumbai", "weather": "The weather in Mumbai is Cloudy +29°C"}
]}

You respond:
{"step": "OUTPUT", "content": "Delhi: Haze at +38°C. Mumbai: Cloudy at +29°C."}
"""


class AgentResponse(BaseModel):
    step: str = Field(..., description="Either TOOL or OUTPUT only")
    content: Optional[str] = Field(None, description="Final answer (OUTPUT step only)")
    tool: Optional[str] = Field(None, description="Tool name (TOOL step only)")
    cities: Optional[List[str]] = Field(None, description="List of cities (TOOL step only)")


def run_agent(user_query: str):
    message_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    tool_called = False
    max_iterations = 8

    for iteration in range(max_iterations):
        print(f"[iter {iteration + 1}] calling model...")

        response = client.chat.completions.parse(
            model="llama3.1:8b",
            response_format=AgentResponse,
            messages=message_history
        )

        raw_result = response.choices[0].message.content
        parsed: AgentResponse = response.choices[0].message.parsed

        print(f"[iter {iteration + 1}] raw: {raw_result}")

        if parsed is None:
            print("⚠️  Could not parse model response.")
            break

        if parsed.step == "OBSERVE":
            print("⚠️  Model tried to output OBSERVE — ignoring and re-prompting.")
            message_history.append({
                "role": "user",
                "content": "OBSERVE is a system-only message. You must now output either TOOL or OUTPUT as your step."
            })
            continue

        message_history.append({"role": "assistant", "content": raw_result})

        if parsed.step == "TOOL":
            if tool_called:
                print("⚠️  Tool already called. Forcing OUTPUT.")
                message_history.append({
                    "role": "user",
                    "content": "You already fetched the weather. Now output the OUTPUT step using the results provided."
                })
                continue

            if parsed.tool not in available_tools:
                print(f"⚠️  Unknown tool: {parsed.tool}")
                break

            if not parsed.cities:
                print("⚠️  No cities provided.")
                break

            results = []
            for city in parsed.cities:
                result = available_tools[parsed.tool](city)
                print(f"🔧  {parsed.tool}({city}) → {result}")
                results.append({"city": city, "weather": result})

            tool_called = True

            message_history.append({
                "role": "user",
                "content": (
                    "Here are the weather results:\n"
                    + json.dumps({"results": results})
                    + "\nNow respond with the OUTPUT step using ONLY these exact values."
                )
            })
            continue

        if parsed.step == "OUTPUT":
            print(f"\n✅  {parsed.content}\n")
            return

        print(f"⚠️  Unknown step '{parsed.step}' — skipping.")

    print("⚠️  Agent did not produce OUTPUT within max iterations.")


if __name__ == "__main__":
    print("\n🌤️  Weather Agent Ready\n")
    user_query = input("Ask about weather → ").strip()
    if user_query:
        run_agent(user_query)
    else:
        print("⚠️  No input provided.")