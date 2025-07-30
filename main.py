from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


loader = PyPDFLoader("docs/wisata_jember.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="db")
vectordb.persist()
retriever = vectordb.as_retriever()

llm = Ollama(model="mistral")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

print("==== Chatbot Wisata Jember ====\n(Ketik 'exit' untuk keluar)\n")

while True:
    query = input("Pertanyaanmu: ")
    if query.lower() in ["exit", "quit"]:
        print("Sampai jumpa!")
        break
    result = qa_chain.run(query)
    print("Jawaban:", result)
    print("\n" + "-"*40 + "\n")