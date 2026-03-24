from ollama import Client
import requests

client = Client(host="http://localhost:11434")



def get_weather(city: str ):
    url =f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200 :
        return f"The weather in {city} is {response.text}"
    else :
        return "something went wrong."


def main():
    user_query = input("-> ")

    response = client.chat(
        model="gemma:2b",
        messages=[
            {
                "role": "user",
                "content": user_query
            }
        ]
    )

    print("🤖 : " + response["message"]["content"])

main()
print(get_weather("delhi"))