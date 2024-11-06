from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
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


class JokeFlow(Workflow):
    llm = OpenAI()

    @step
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic

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
    result = await w.run(topic="pirates")
    print(result)
    draw_most_recent_execution(w, filename="joke_flow_recent.html")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
