from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")

async def main():
    w = MyWorkflow(timeout=1, verbose=True)
    result = await w.run()
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


