import json
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Sequence,
    Tuple, Optional,
)

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ToolCallSummaryMessage, ChatMessage, AgentEvent, ThoughtEvent, BaseAgentEvent, BaseChatMessage
from autogen_core import RoutedAgent, MessageContext, message_handler, FunctionCall, CancellationToken
from autogen_core.models import FunctionExecutionResult, CreateResult, AssistantMessage, RequestUsage
from autogen_core.tools import Workbench
from pydantic.dataclasses import dataclass


@dataclass
class MyMessageType:
    content: str


class ExperimentAgent(AssistantAgent):
    """Agent responsible for analyzing data and results."""
    def __init__(
            self,
            name: str = "Analysis_Agent",
            system_message: Optional[str] = None,
            tools: Optional[List[Callable]] = None,
            **kwargs
    ):
        default_system_message = """You are an Experiment Agent specialized in designing, implementing, 
        and running experiments involving Large Language Models. Your expertise lies in experimental design, 
        controlling variables, and ensuring reproducibility. You can propose experimental setups, 
        implement evaluation metrics, and execute tests to gather data on LLM performance and behavior.
        When designing experiments, focus on clarity, measurability, and scientific rigor.
        """

        AssistantAgent.__init__(
            self,
            name=name,
            system_message=system_message or default_system_message,
            tools=tools,
            **kwargs
        )

    async def on_messages_stream(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
        """
        Process the incoming messages with the assistant agent and yield events/responses as they happen.
        """

        # Gather all relevant state here
        agent_name = self.name
        model_context = self._model_context
        memory = self._memory
        system_messages = self._system_messages
        workbench = self._workbench
        handoff_tools = self._handoff_tools
        handoffs = self._handoffs
        model_client = self._model_client
        model_client_stream = self._model_client_stream
        reflect_on_tool_use = self._reflect_on_tool_use
        tool_call_summary_format = self._tool_call_summary_format
        output_content_type = self._output_content_type
        format_string = self._output_content_type_format


        # STEP 1: Add new user/handoff messages to the model context
        await self._add_messages_to_context(
            model_context=model_context,
            messages=messages,
        )

        # STEP 2: Update model context with any relevant memory
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        for event_msg in await self._update_model_context_with_memory(
            memory=memory,
            model_context=model_context,
            agent_name=agent_name,
        ):
            inner_messages.append(event_msg)
            yield event_msg

        # STEP 3: Run the first inference
        model_result = None
        async for inference_output in self._call_llm(
                model_client=model_client,
                model_client_stream=model_client_stream,
                system_messages=system_messages,
                model_context=model_context,
                workbench=workbench,
                handoff_tools=handoff_tools,
                agent_name=agent_name,
                cancellation_token=cancellation_token,
                output_content_type=output_content_type,
        ):
            if isinstance(inference_output, CreateResult):
                model_result = inference_output
            else:
                # Streaming chunk event
                yield inference_output

        assert model_result is not None, "No model result was produced."

        # --- NEW: If the model produced a hidden "thought," yield it as an event ---
        if model_result.thought:
            thought_event = ThoughtEvent(content=model_result.thought, source=agent_name)
            yield thought_event
            inner_messages.append(thought_event)

        # Add the assistant message to the model context (including thought if present)
        await model_context.add_message(
            AssistantMessage(
                content=model_result.content,
                source=agent_name,
                thought=getattr(model_result, "thought", None),
            )
        )

        # STEP 4: Process the model output
        async for output_event in self._process_model_result(
                model_result=model_result,
                inner_messages=inner_messages,
                cancellation_token=cancellation_token,
                agent_name=agent_name,
                system_messages=system_messages,
                model_context=model_context,
                workbench=workbench,
                handoff_tools=handoff_tools,
                handoffs=handoffs,
                model_client=model_client,
                model_client_stream=model_client_stream,
                reflect_on_tool_use=reflect_on_tool_use,
                tool_call_summary_format=tool_call_summary_format,
                output_content_type=output_content_type,
                format_string=format_string,
        ):
            yield output_event