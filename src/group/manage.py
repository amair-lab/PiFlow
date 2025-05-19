import asyncio
from typing import List, Sequence, AsyncGenerator, Optional

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.base import ChatAgent, TerminationCondition, TaskResult
from autogen_agentchat.messages import TextMessage, BaseChatMessage, ModelClientStreamingChunkEvent, BaseAgentEvent, StopMessage, MessageFactory
from autogen_agentchat.teams._group_chat._events import GroupChatStart, GroupChatTermination, SerializableException
from autogen_agentchat.teams._group_chat._selector_group_chat import SelectorGroupChat, SelectorGroupChatConfig, SelectorGroupChatManager, SelectorFuncType, CandidateFuncType
from autogen_core import CancellationToken, AgentId, SingleThreadedAgentRuntime, AgentRuntime
from autogen_core.models import ChatCompletionClient

class HypoValidManager(SelectorGroupChatManager):
    def __init__(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        message_factory: MessageFactory,
        model_client: ChatCompletionClient,
        selector_prompt: str,
        allow_repeated_speaker: bool,
        selector_func: Optional[SelectorFuncType],
        max_selector_attempts: int,
        candidate_func: Optional[CandidateFuncType],
        emit_team_events: bool,
        model_client_streaming: bool = False,
    ):
        super().__init__(
            name=name,
            group_topic_type=group_topic_type,
            output_topic_type=output_topic_type,
            participant_topic_types=participant_topic_types,
            participant_names=participant_names,
            participant_descriptions=participant_descriptions,
            output_message_queue=output_message_queue,
            termination_condition=termination_condition,
            max_turns=max_turns,
            message_factory=message_factory,
            emit_team_events=emit_team_events,
            selector_prompt=selector_prompt,
            model_client=model_client,
            allow_repeated_speaker=allow_repeated_speaker,
            selector_func=selector_func,
            max_selector_attempts=max_selector_attempts,
            candidate_func=candidate_func
        )

        self._model_client: OpenAIChatCompletionClient | ChatCompletionClient = model_client
        self._selector_prompt = selector_prompt
        self._previous_speaker: str | None = None
        self._allow_repeated_speaker = allow_repeated_speaker
        self._selector_func = selector_func
        self._max_selector_attempts = max_selector_attempts


class HypoValidGroupChat(SelectorGroupChat):
    component_config_schema = SelectorGroupChatConfig
    component_provider_override = "autogen_agentchat.teams.SelectorGroupChat"

    def __init__(
            self,
            participants: List[ChatAgent],
            model_client: ChatCompletionClient,
            note_taker_output_file: str,
            *,
            termination_condition: TerminationCondition | None = None,
            max_turns: int | None = None,
            runtime: AgentRuntime | None = None,
            selector_prompt: str = """You are in a role play game. The following roles are available:
            {roles}.
            Read the following conversation. Then select the next role from {participants} to play. Only return the role.

            {history}

            Read the above conversation. Then select the next role from {participants} to play. Only return the role.
            """,
            allow_repeated_speaker: bool = False,
            max_selector_attempts: int = 3,
            selector_func: Optional[SelectorFuncType] = None,
            candidate_func: Optional[CandidateFuncType] = None,
            custom_message_types: List[type[BaseAgentEvent | BaseChatMessage]] | None = None,
            emit_team_events: bool = False,
            model_client_streaming: bool = False,
    ):
        # Call super().__init__() with all the parameters SelectorGroupChat expects
        SelectorGroupChat.__init__(
            self,
            participants=participants,
            model_client=model_client,
            termination_condition=termination_condition,
            max_turns=max_turns,
            selector_prompt=selector_prompt,
            allow_repeated_speaker=allow_repeated_speaker,
            max_selector_attempts=max_selector_attempts,
            selector_func=selector_func,
            candidate_func=candidate_func,
            custom_message_types=custom_message_types,
            emit_team_events=emit_team_events,
            model_client_streaming=model_client_streaming,
            runtime=runtime,
        )
        # Validate the participants.
        if len(participants) < 2:
            raise ValueError("At least two participants are required for SelectorGroupChat.")
        self._selector_prompt = selector_prompt
        self._model_client = model_client
        self._allow_repeated_speaker = allow_repeated_speaker
        self._selector_func = selector_func
        self._max_selector_attempts = max_selector_attempts
        self._candidate_func = candidate_func
        self._model_client_streaming = model_client_streaming

    async def run_stream(
        self,
        *,
        task: str | BaseChatMessage | Sequence[BaseChatMessage] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | TaskResult, None]:
        """Run the team and produces a stream of messages and the final result
        of the type :class:`~autogen_agentchat.base.TaskResult` as the last item in the stream. Once the
        team is stopped, the termination condition is reset.

        .. note::

            If an agent produces :class:`~autogen_agentchat.messages.ModelClientStreamingChunkEvent`,
            the message will be yielded in the stream but it will not be included in the
            :attr:`~autogen_agentchat.base.TaskResult.messages`.

        Args:
            task (str | BaseChatMessage | Sequence[BaseChatMessage] | None): The task to run the team with. Can be a string, a single :class:`BaseChatMessage` , or a list of :class:`BaseChatMessage`.
            cancellation_token (CancellationToken | None): The cancellation token to kill the task immediately.
                Setting the cancellation token potentially put the team in an inconsistent state,
                and it may not reset the termination condition.
                To gracefully stop the team, use :class:`~autogen_agentchat.conditions.ExternalTermination` instead.

        Returns:
            stream: an :class:`~collections.abc.AsyncGenerator` that yields :class:`~autogen_agentchat.messages.BaseAgentEvent`, :class:`~autogen_agentchat.messages.BaseChatMessage`, and the final result :class:`~autogen_agentchat.base.TaskResult` as the last item in the stream.

        Example using the :class:`~autogen_agentchat.teams.RoundRobinGroupChat` team:

        .. code-block:: python

            import asyncio
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                agent1 = AssistantAgent("Assistant1", model_client=model_client)
                agent2 = AssistantAgent("Assistant2", model_client=model_client)
                termination = MaxMessageTermination(3)
                team = RoundRobinGroupChat([agent1, agent2], termination_condition=termination)

                stream = team.run_stream(task="Count from 1 to 10, respond one at a time.")
                async for message in stream:
                    print(message)

                # Run the team again without a task to continue the previous task.
                stream = team.run_stream()
                async for message in stream:
                    print(message)


            asyncio.run(main())


        Example using the :class:`~autogen_core.CancellationToken` to cancel the task:

        .. code-block:: python

            import asyncio
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.ui import Console
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_core import CancellationToken
            from autogen_ext.models.openai import OpenAIChatCompletionClient


            async def main() -> None:
                model_client = OpenAIChatCompletionClient(model="gpt-4o")

                agent1 = AssistantAgent("Assistant1", model_client=model_client)
                agent2 = AssistantAgent("Assistant2", model_client=model_client)
                termination = MaxMessageTermination(3)
                team = RoundRobinGroupChat([agent1, agent2], termination_condition=termination)

                cancellation_token = CancellationToken()

                # Create a task to run the team in the background.
                run_task = asyncio.create_task(
                    Console(
                        team.run_stream(
                            task="Count from 1 to 10, respond one at a time.",
                            cancellation_token=cancellation_token,
                        )
                    )
                )

                # Wait for 1 second and then cancel the task.
                await asyncio.sleep(1)
                cancellation_token.cancel()

                # This will raise a cancellation error.
                await run_task


            asyncio.run(main())

        """

        # Create the messages list if the task is a string or a chat message.
        messages: List[BaseChatMessage] | None = None
        if task is None:
            pass
        elif isinstance(task, str):
            messages = [TextMessage(content=task, source="user")]
        elif isinstance(task, BaseChatMessage):
            messages = [task]
        elif isinstance(task, list):
            if not task:
                raise ValueError("Task list cannot be empty.")
            messages = []
            for msg in task:
                if not isinstance(msg, BaseChatMessage):
                    raise ValueError("All messages in task list must be valid BaseChatMessage types")
                messages.append(msg)
        else:
            raise ValueError("Task must be a string, a BaseChatMessage, or a list of BaseChatMessage.")
        # Check if the messages types are registered with the message factory.
        if messages is not None:
            for msg in messages:
                if not self._message_factory.is_registered(msg.__class__):
                    raise ValueError(
                        f"Message type {msg.__class__} is not registered with the message factory. "
                        "Please register it with the message factory by adding it to the "
                        "custom_message_types list when creating the team."
                    )

        if self._is_running:
            raise ValueError("The team is already running, it cannot run again until it is stopped.")
        self._is_running = True

        if self._embedded_runtime:
            # Start the embedded runtime.
            assert isinstance(self._runtime, SingleThreadedAgentRuntime)
            self._runtime.start()

        if not self._initialized:
            await self._init(self._runtime)

        shutdown_task: asyncio.Task[None] | None = None
        if self._embedded_runtime:

            async def stop_runtime() -> None:
                assert isinstance(self._runtime, SingleThreadedAgentRuntime)
                try:
                    # This will propagate any exceptions raised.
                    await self._runtime.stop_when_idle()
                    # Put a termination message in the queue to indicate that the group chat is stopped for whatever reason
                    # but not due to an exception.
                    await self._output_message_queue.put(
                        GroupChatTermination(
                            message=StopMessage(
                                content="The group chat is stopped.", source=self._group_chat_manager_name
                            )
                        )
                    )
                except Exception as e:
                    # Stop the consumption of messages and end the stream.
                    # NOTE: we also need to put a GroupChatTermination event here because when the runtime
                    # has an exception, the group chat manager may not be able to put a GroupChatTermination event in the queue.
                    # This may not be necessary if the group chat manager is able to handle the exception and put the event in the queue.
                    await self._output_message_queue.put(
                        GroupChatTermination(
                            message=StopMessage(
                                content="An exception occurred in the runtime.", source=self._group_chat_manager_name
                            ),
                            error=SerializableException.from_exception(e),
                        )
                    )

            # Create a background task to stop the runtime when the group chat
            # is stopped or has an exception.
            shutdown_task = asyncio.create_task(stop_runtime())

        try:
            await self._runtime.send_message(
                GroupChatStart(messages=messages),
                recipient=AgentId(type=self._group_chat_manager_topic_type, key=self._team_id),
                cancellation_token=cancellation_token,
            )
            # Collect the output messages in order.
            output_messages: List[BaseAgentEvent | BaseChatMessage] = []
            stop_reason: str | None = None
            # Yield the messages until the queue is empty.
            while True:
                message_future = asyncio.ensure_future(self._output_message_queue.get())
                if cancellation_token is not None:
                    await cancellation_token.link_future(message_future)
                # Wait for the next message, this will raise an exception if the task is cancelled.
                message = await message_future

                if isinstance(message, GroupChatTermination):
                    # If the message contains an error, we need to raise it here.
                    # This will stop the team and propagate the error.
                    if message.error is not None:
                        raise RuntimeError(str(message.error))
                    stop_reason = message.message.content
                    break
                yield message
                if isinstance(message, ModelClientStreamingChunkEvent):
                    # Skip the model client streaming chunk events.
                    continue
                output_messages.append(message)

            # Yield the final result.
            result = TaskResult(messages=output_messages, stop_reason=stop_reason)

            yield result

        finally:
            try:
                if shutdown_task is not None:
                    # Wait for the shutdown task to finish.
                    # This will propagate any exceptions raised.
                    await shutdown_task
            finally:
                # Clear the output message queue.
                while not self._output_message_queue.empty():
                    self._output_message_queue.get_nowait()

                # Indicate that the team is no longer running.
                self._is_running = False
