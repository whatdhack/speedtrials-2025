from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
import os
import glob

load_dotenv(dotenv_path="/mnt/e/thegpvc/speed.env")


# Load environment variables (API keys, etc.)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the directory containing your CSV files
csv_directory = "/mnt/e/thegpvc/speedtrials-2025-whatdhack/data"  # Change this to your directory

# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))

# Check if any CSV files were found
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {csv_directory}")

# Load data from all CSV files
documents = []
startdoc=4
for file in csv_files[startdoc:(startdoc+1)]:
    print (f" loading file {file}")
    loader = CSVLoader(file_path=file)
    documents.extend(loader.load())

#print (f" documents {documents}")

embeddings = OpenAIEmbeddings()
# Create a vectorstore index from the combined documents
index = VectorstoreIndexCreator(embedding=embeddings).from_documents(documents)
retriever=index.vectorstore.as_retriever(search_kwargs={"k": len(documents)})


def deprecated_lccbot():
    # Create a question-answering chain
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=retriever,
    )
    
    '''
    # Example questions
    questions = [
        "What are the main topics discussed in these files?",
        "Can you summarize the information about [specific topic] across all files?",
        "Give me a list of all the unique identifiers mentioned in the files.",
        "How are these files related to each other?"
    ]
    
    # Perform queries and print results
    for question in questions:
        result = chain.run(question)
        print(f"Question: {question}")
        print(f"Answer: {result}")
        print("-" * 20)
    '''
    
    while True:
        question = input('Your question (e.g. how many violations are there ?) ')
        if question.lower() in ['quit', 'exit', 'bye']:
            print('Goodbye!')
            break
        result = chain.run(question)
        print(f"Question: {question}")
        print(f"Answer: {result}")
        print("-" * 20)


def lccbot():

    llm = ChatOpenAI()
    
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    #query = "who are you ?"    
    #reply = chain.invoke({"input": query})
    #print (f" reply {reply}")
    while True:
        question = input('Your question (e.g. how many violations are there ?) ')
        if question.lower() in ['quit', 'exit', 'bye']:
            print('Goodbye!')
            break
        result = chain.invoke({"input": question})
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print("-" * 20)


#deprecated_lccbot()
lccbot()

