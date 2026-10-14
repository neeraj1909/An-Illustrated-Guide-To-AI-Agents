from llm import LLM


class Memory:
    """Simple memory module to store conversation history."""
    
    def __init__(self):
        self.messages = []
        
    def add(
        self, role: str, content: str, tool_call: dict | None = None, **kwargs
    ) -> None:
        """Add a message to memory."""
        message = {"role": role, "content": content}
        
        # Tool call
        if tool_call:
            message["tool_calls"] = [tool_call]
            
        # Append message to memory
        self.messages.append(message)
        
    def get_messages(self) -> list[dict]:
        """Get all messages."""
        return self.messages


class TrimmingMemory(Memory):
    """Memory that keeps only the last two user/assistant turns."""
    def add(self, role: str, content: str, **kwargs) -> None:
        # Add the new message first using the parent class (Memory)
        super().add(role, content, **kwargs)
        
        # keep system message with most recent two turns (4 messages)
        system = [
            message for message in self.messages if message["role"] == "system"
        ]
        
        turns = [
            message for message in self.messages if message["role"] != "system"
        ]
        
        self.messages = system + turns[-4:]
                    