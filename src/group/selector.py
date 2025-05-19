from typing import Sequence

from autogen_agentchat.messages import ChatMessage, TextMessage, AgentEvent
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.cache import ChatCompletionCache
from autogen_ext.models.openai import OpenAIChatCompletionClient


class Selector:
    def __init__(self, model_client: OpenAIChatCompletionClient | ChatCompletionCache):
        self._client = model_client
        self._select_prompt = lambda speaking_order: [
            SystemMessage(content="""
There is a scientific task that needs many agents to deal with. They are in different functions and backgrounds. See the speaking order currently, then select an agent from participants to perform the next speaking.

Note: **Only select one agent.**
"""),
            UserMessage(content=speaking_order, source="user"),
        ]

    async def select_on__speaking_order(self, messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
        """
            Select a speaker from the chatting order.
        """
        _speaking_order =  []

        for message in messages:
            print(message)
            _speaking_order.append(message.source)

        if _speaking_order:
            respond = await self._client.create(
                messages=self._select_prompt(_speaking_order)
            )
            selection = respond.content.strip()
            if selection not in set (_speaking_order):
                selection = None
        else:
            selection = None

        return selection