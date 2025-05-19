from typing import (
    Callable,
    List,
    Optional,
)

from colorama import Fore, Style, init

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import message_handler, MessageContext, RoutedAgent

init(autoreset=True)

class HypothesisAgent(AssistantAgent):
    """Agent responsible for analyzing data and results."""
    def __init__(
            self,
            name: str = "Analysis_Agent",
            system_message: Optional[str] = None,
            tools: Optional[List[Callable]] = None,
            **kwargs
    ):
        default_system_message = """You are a Hypothesis Agent specialized in formulating, refining, and testing 
        scientific hypotheses about Large Language Models. Your expertise lies in connecting theoretical frameworks 
        with empirical observations, identifying potential causal relationships, and proposing testable predictions.
        You excel at critical thinking, maintaining scientific rigor, and adjusting hypotheses based on new evidence.
        Focus on clarity, falsifiability, and scientific value when generating hypotheses."""

        super().__init__(
            name=name,
            system_message=system_message or default_system_message,
            tools=tools,
            **kwargs
        )
