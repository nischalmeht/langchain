# Requires OPENAI_API_KEY in the environment (or a .env file)
# pip install youtube-transcript-api langchain-community langchain-openai faiss-cpu tiktoken python-dotenv
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

load_dotenv()

# Step 1a - Indexing (Document Ingestion)
video_id = "Gfr50f6ZBvo"  # only the ID, not full URL
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
except TranscriptsDisabled:
    raise SystemExit("No captions available for this video.")

# Step 1b - Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Step 3 - Augmentation
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=["context", "question"],
)


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# Step 4 - Generation, step by step
question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)
final_prompt = prompt.invoke({"context": format_docs(retrieved_docs), "question": question})
answer = llm.invoke(final_prompt)
print(answer.content)

# Building a Chain
parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)
main_chain = parallel_chain | prompt | llm | StrOutputParser()

print(main_chain.invoke("who is Demis"))
print(main_chain.invoke("Can you summarize the video"))
