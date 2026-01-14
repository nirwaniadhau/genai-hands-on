from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader(r"D:\Gen_Ai\LangChain_Document_Loaders\07_CIAS_ASSIGNMENT01.pdf")
docs=loader.load()
print(len(docs))
print(docs[0])

