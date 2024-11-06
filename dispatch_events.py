from llama_index.core.workflow import (
    step,
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow
)


class MyEvent(Event):
    pass


class MyEventResult(Event):
    result: str


class GatherEvent(Event):
    pass


class MyWorkflow(Workflow):
    @step
    async def dispatch_step(
        self, ctx: Context, ev: StartEvent
    ) -> MyEvent | GatherEvent:
        ctx.send_event(MyEvent())
        ctx.send_event(MyEvent())

        return GatherEvent()

    @step
    async def handle_my_event(self, ev: MyEvent) -> MyEventResult:
        return MyEventResult(result="result")

    @step
    async def gather(
        self, ctx: Context, ev: GatherEvent | MyEventResult
    ) -> StopEvent | None:
        # wait for events to finish
        events = ctx.collect_events(ev, [MyEventResult, MyEventResult])
        if not events:
            return None

        return StopEvent(result=events)


async def main():
    w = MyWorkflow(timeout=10, verbose=True)
    result = await w.run()

    from pprint import pprint
    pprint(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

