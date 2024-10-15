RAG adds your data to LLMs. Stages:
* **Loading:** getting data from where it lives: PDF, DB, API and etc. 
* **Indexing:** creating `vector embeddings` numerical representations of the meaning of your data. 
* **Storing:** storing indexed data with its metadata
* **Querying:** involves sub-querying, multi-step querying and hybrid strategies. 
* **Evaluation:** measuring how accurate, faithful and fast responses are. 

Terms encountered within these 3 stages:
* **Loading**
  * `Document` is container around any data source. `Node` is atomic unit of data. 
  * `Connectors` (Readers) ingests data from different sources and data formats into `Documents` and `Nodes`. 
* **Indexing**
  * Generating `vector embeddings` and storing them into `vector store`. 
  * Queries are also converted into embeddings and vector store will find data that is numerically similar to query embedding. 
* **Querying**
  * `Retrieval` defines strategy to efficiently retrieve relevant context from index. 
  * `Router` determines which retriever will be used. Specifically, `RouterRetrieval` class is responsible for selecting one or multiple retrievals to execute a query. 
  * `Node Postprocessor` takes in a set of retrieved nodes and applies transformations, filtering and re-ranking logic to them. 
  * `Response Synthesizer` generates response from LLM using user query and given set of retrieved text chunks. 
