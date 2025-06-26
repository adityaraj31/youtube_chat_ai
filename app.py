import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Step 1a - Indexing (Document Ingestion)
video_id = "Gfr50f6ZBvo"  # only the ID, not full URL

try:
    # Get transcript
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    # Flatten to plain text  ...
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

    # Step 1b - Indexing (Text Splitting)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    print(f"Number of chunks created: {len(chunks)}")
    if len(chunks) > 0:
        print("First chunk sample:", chunks[0])

    # Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Step 2 - Retrieval
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Step 3 - Augmentation
    llm = ChatGroq(
        model_name="llama3-8b-8192", 
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")  # Get API key from environment variables
    )

    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )

    question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.invoke({"context": context_text, "question": question})

    # Step 4 - Generation
    answer = llm.invoke(final_prompt)
    print(answer.content)

except TranscriptsDisabled:
    print("No captions available for this video.")
except IndexError:
    print("Not enough chunks were created from the transcript.")
except Exception as e:
    print(f"An error occurred: {str(e)}")