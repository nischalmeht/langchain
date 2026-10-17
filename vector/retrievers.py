# Requires OPENAI_API_KEY in the environment (or a .env file)
# pip install langchain chromadb faiss-cpu openai tiktoken langchain_openai langchain-community wikipedia
from dotenv import load_dotenv
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def print_results(results):
    for i, doc in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(doc.page_content)


# ---------------------------------------------------------------- Wikipedia Retriever
retriever = WikipediaRetriever(top_k_results=2, lang="en")
query = "the geopolitical history of india and pakistan from the perspective of a chinese"
docs = retriever.invoke(query)
print_results(docs)

# ---------------------------------------------------------------- Vector Store Retriever
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding_model = OpenAIEmbeddings()

# in-memory Chroma store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
query = "What is Chroma used for?"
print_results(retriever.invoke(query))

# same search directly on the vector store
print_results(vectorstore.similarity_search(query, k=2))

# ---------------------------------------------------------------- MMR
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vectorstore = FAISS.from_documents(documents=docs, embedding=embedding_model)

# k = top results, lambda_mult = relevance-diversity balance
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5},
)
print_results(retriever.invoke("What is langchain?"))

# ---------------------------------------------------------------- Multiquery Retriever
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

vectorstore = FAISS.from_documents(documents=all_docs, embedding=embedding_model)

similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
)

query = "How to improve energy levels and maintain balance?"
print_results(similarity_retriever.invoke(query))
print("*" * 150)
print_results(multiquery_retriever.invoke(query))

# ---------------------------------------------------------------- ContextualCompressionRetriever
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),
    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),
    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),
    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"}),
]

vectorstore = FAISS.from_documents(docs, embedding_model)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# the LLM extracts only the parts of each document relevant to the query
compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-3.5-turbo"))
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)
print_results(compression_retriever.invoke("What is photosynthesis?"))
