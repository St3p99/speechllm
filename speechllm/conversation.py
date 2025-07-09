import dataclasses
from enum import Enum, auto
from typing import List

class SeparatorStyle(Enum):
    """Different separator style."""
    MPT = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.MPT
    sep: str = "<|eot_id|>"
    sep2: str = None
    version: str = "llama_3_1"

    skip_next: bool = False

    def get_prompt(self):
        # Only handle MPT (llama_3) style, no image
        if self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, *_ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_llama_3_1 = Conversation(
    system="""<|start_header_id|>system<|end_header_id|>\n\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""",
    roles=[
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ],
    version="llama_3_1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
)

default_conversation = conv_llama_3_1
conv_templates = {
    "llama_3_1": conv_llama_3_1,
}

if __name__ == "__main__":
    conv = conv_llama_3_1.copy()
    conv.append_message(conv.roles[0], "Hello, how are you?")
    print(conv.get_prompt())
