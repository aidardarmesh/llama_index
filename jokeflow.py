from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)

# `pip install llama-index-llms-openai` if you don't already have it
from llama_index.llms.openai import OpenAI


class JokeEvent(Event):
    joke: str


class ProgressEvent(Event):
    msg: str


class JokeFlow(Workflow):
    llm = OpenAI()

    @step
    async def generate_joke(self, ctx: Context, ev: StartEvent) -> JokeEvent:
        topic = ev.topic
        ctx.write_event_to_stream(ProgressEvent(msg="Joke generation is happening"))
        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        print(response)
        return JokeEvent(joke=str(response))

    @step
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))


async def main():
    draw_all_possible_flows(JokeFlow, filename="joke_flow_all.html")
    w = JokeFlow(timeout=60, verbose=False)
    handler = w.run(topic="pirates")
    async for event in handler.stream_events():
        print(event)
    result = await handler
    print(str(result))
    draw_most_recent_execution(w, filename="joke_flow_recent.html")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
