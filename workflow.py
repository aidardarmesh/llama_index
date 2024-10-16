from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
from llama_index.utils.workflow import draw_all_possible_flows


class FirstEvent(Event):
    first_output: str


class SecondEvent(Event):
    second_output: str


class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> FirstEvent:
        print(ev.first_input)
        return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        print(ev.first_output)
        return SecondEvent(second_output="Second step complete.")

    @step
    async def step_three(self, ev: SecondEvent) -> StopEvent:
        print(ev.second_output)
        return StopEvent(result="Workflow complete.")


async def main():
    w = MyWorkflow(timeout=1, verbose=True)
    result = await w.run(first_input="Start the workflow.")
    print(result)
    draw_all_possible_flows(w, filename="multi_step_workflow.html")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


