import json
import urllib.request

from agent import TinyAgent


# agent = TinyAgent()
# output = agent.run("What is 2 + 2?")
# print(f"{output}")


# prepare request
data = json.dumps(
    {
        "model": "gemma3:4b",
        "messages": [{"role": "user", "content": "Hi! How's life?"}]
    }
).encode("utf-8")

# post request
req = urllib.request.Request(
    url="http://localhost:11434/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"}
)

# parse response
with urllib.request.urlopen(req) as response:
    result = json.loads(response.read())
    
# print response
print(result["choices"][0])
