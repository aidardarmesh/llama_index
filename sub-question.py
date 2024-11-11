import os, json
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.utils.workflow import draw_all_possible_flows
from dotenv import load_dotenv

load_dotenv()

class QueryEvent(Event):
    question: str


class AnswerEvent(Event):
    question: str
    answer: str


class SubQuestionQueryEngine(Workflow):
    @step
    async def query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        if hasattr(ev, "query"):
            await ctx.set("original_query", ev.query)
            print(f"Query is {await ctx.get('original_query')}")

        if hasattr(ev, "llm"):
            await ctx.set("llm", ev.llm)

        if hasattr(ev, "tools"):
            await ctx.set("tools", ev.tools)

        response = (await ctx.get("llm")).complete(
            f"""
            Given a user question, and a list of tools, output a list of
            relevant sub-questions, such that the answers to all the
            sub-questions put together will answer the question. Respond
            in pure JSON without any markdown, like this:
            {{
                "sub_questions": [
                    "What is the population of San Francisco?",
                    "What is the budget of San Francisco?",
                    "What is the GDP of San Francisco?"
                ]
            }}
            Here is the user question: {await ctx.get('original_query')}

            And here is the list of tools: {await ctx.get('tools')}
            """
        )
        print(f"Sub-questions are: {response}")

        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]

        await ctx.set("sub_questions_count", len(sub_questions))

        for question in sub_questions:
            self.send_event(QueryEvent(question=question))

    @step
    async def sub_question(self, ctx: Context, ev: QueryEvent) -> AnswerEvent:
        print(f"Sub-question is {ev.question}")

        agent = ReActAgent.from_tools(
            await ctx.get("tools"), llm=await ctx.get("llm"), verbose=True, max_iterations=100
        )
        response = agent.chat(ev.question)

        return AnswerEvent(question=ev.question, answer=str(response))

    @step
    async def combine_answers(self, ctx: Context, ev: AnswerEvent) -> StopEvent | None:
        ready = ctx.collect_events(ev, [AnswerEvent] * await ctx.get("sub_questions_count"))
        if ready is None:
            return

        answers = "\n\n".join(
            [
                f"Question: {event.question} \n Answer: {ev.answer}"
                for event in ready
            ]
        )

        prompt = f"""
            You are given an overall question that has been split into sub-questions,
            each of which has been answered. Combine the answers to all the sub-questions
            into a single answer to the original question.

            Original question: {await ctx.get('original_query')}

            Sub-questions and answers:
            {answers}
        """

        print(f"Final prompt is {prompt}")

        response = (await ctx.get("llm")).complete(prompt)

        print(f"Final response is", response)

        return StopEvent(result=str(response))


draw_all_possible_flows(
    SubQuestionQueryEngine, filename="sub_question_query_engine.html"
)

folder = "./data/sf_budgets/"
files = os.listdir(folder)

query_engine_tools = []
for file in files:
    year = file.split(" - ")[0]
    index_persist_path = f"./storage/budget-{year}/"

    if os.path.exists(index_persist_path):
        storage_context = StorageContext.from_defaults(
            persist_dir=index_persist_path
        )
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader(
            input_files=[folder + file]
        ).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(index_persist_path)

    engine = index.as_query_engine()
    query_engine_tools.append(
        QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name=f"budget_{year}",
                description=f"Information about San Francisco's budget in {year}",
            ),
        )
    )


async def main():
    engine = SubQuestionQueryEngine(timeout=200, verbose=True)
    llm = OpenAI(model="gpt-4o")
    result = await engine.run(
        llm=llm,
        tools=query_engine_tools,
        query="How has the total amount of San Francisco's budget changed from 2016 to 2023?",
    )

    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

