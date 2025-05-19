import logging
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    List,
    Sequence,
    Tuple, Optional, Union,
)

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import HandoffMessage, AgentEvent, ChatMessage, MemoryQueryEvent, ModelClientStreamingChunkEvent, TextMessage, ToolCallRequestEvent, ToolCallExecutionEvent, ToolCallSummaryMessage, ThoughtEvent, BaseAgentEvent, BaseChatMessage
from autogen_core import FunctionCall, CancellationToken, EVENT_LOGGER_NAME
from autogen_core.model_context import ChatCompletionContext, BufferedChatCompletionContext
from autogen_core.models import UserMessage, CreateResult, AssistantMessage, FunctionExecutionResultMessage, FunctionExecutionResult, LLMMessage, RequestUsage, SystemMessage, ChatCompletionClient
from autogen_core.tools import Workbench, BaseTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from src.group.workflow import PrincipleFlow

event_logger = logging.getLogger(EVENT_LOGGER_NAME)


class Planner(AssistantAgent):
    """Agent responsible for analyzing data and results."""
    def __init__(
            self,
            name: str = "Planner_Agent",
            system_message: Optional[str] = None,
            tools: Optional[List[Callable]] = None,
            model_client: OpenAIChatCompletionClient | ChatCompletionClient = None,
            flow: PrincipleFlow = None,
            is_sas: bool = False,
            is_mas: bool = False,
            is_principled: bool = False,
            is_prompted: bool = False,
            **kwargs
    ):
        default_system_message = "You plan the task by decoupling and assignment. "

        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message or default_system_message,
            tools=tools,
            **kwargs
        )

        # is equal to OpenAIChatCompletionClient, using `.create(...)` to get LLM response.
        self.llm_client = model_client

        self.flow: PrincipleFlow = flow

        # Begin. This is for the config of Planner's behavior.
        self.is_sas: bool = is_sas
        self.is_mas: bool = is_mas
        self.is_principled: bool = is_principled
        self.is_prompted: bool = is_prompted        # used.

        self.processed_messages = set()  # Track which messages have been processed

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


        # [STEP OMNI] FOR **ALL**: Listen for all messages for all baselines & PiFlow.
        await self.flow.listen_messages(messages)

        model_result = None

        # Handle the different reasoning modes
        if self.is_principled and self.is_prompted:
            # Combined mode: Use both PrincipleFlow and LLM
            principled_suggestion = await self.flow.run_principled_reasoning(messages)

            event_logger.info(f"🍁 Suggestion Detected: \n [{principled_suggestion}]")

            # Add the PrincipleFlow suggestion to the context as a special message
            flow_message = UserMessage(
                content=f"""
                PRINCIPLE GUIDANCE: 

                {principled_suggestion}

                Based on the above guidance, provide a synthesized response that incorporates this principle-based suggestion with your own reasoning. Focus on guiding the Hypothesis Agent with clear, specific direction. You do not propose any hypothesis. 
                """,
                source="user"
            )

            # Create a temporary context with the flow message for this inference
            system_messages = list(system_messages)
            await model_context.add_message(flow_message)

            # Call LLM with the enhanced context
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

        elif self.is_principled:
            # PrincipleFlow-only mode
            suggestion = await self.flow.run_principled_reasoning(messages)
            model_result = CreateResult(
                finish_reason="stop",
                content=suggestion,
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
                logprobs=None,
                thought=None
            )

        elif self.is_prompted:
            # LLM-only mode
            # STEP 3: Run the first inference
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

        else:
            # Fallback mode
            model_result = CreateResult(
                finish_reason="stop",
                content="Follow the Hypothesis-Validation workflow, go on.",
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
                logprobs=None,
                thought=None
            )


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

    @classmethod
    async def _call_llm(
        cls,
        model_client: OpenAIChatCompletionClient,
        model_client_stream: bool,
        system_messages: List[SystemMessage],
        model_context: ChatCompletionContext,
        workbench: Workbench,
        handoff_tools: List[BaseTool[Any, Any]],
        agent_name: str,
        cancellation_token: CancellationToken,
        output_content_type: type[BaseModel] | None,
    ) -> AsyncGenerator[Union[CreateResult, ModelClientStreamingChunkEvent], None]:
        """
        Perform a model inference and yield either streaming chunk events or the final CreateResult.
        """
        all_messages = await model_context.get_messages()
        llm_messages = cls._get_compatible_context(model_client=model_client, messages=system_messages + all_messages)

        tools = (await workbench.list_tools()) + handoff_tools

        if model_client_stream:
            model_result: Optional[CreateResult] = None
            async for chunk in model_client.create_stream(
                llm_messages,
                tools=tools,
                json_output=output_content_type,
                cancellation_token=cancellation_token,
            ):
                if isinstance(chunk, CreateResult):
                    model_result = chunk
                elif isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(content=chunk, source=agent_name)
                else:
                    raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
            if model_result is None:
                raise RuntimeError("No final model result in streaming mode.")
            yield model_result
        else:
            model_result = await model_client.create(
                llm_messages,
                tools=tools,
                cancellation_token=cancellation_token,
                json_output=output_content_type,
            )
            yield model_result