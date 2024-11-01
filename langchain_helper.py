from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


embeddings=OpenAIEmbeddings()

def create_vector_db_from_youtube_urls(videos_url: str)-> FAISS:
    loader=YoutubeLoader.from_youtube_url(videos_url)
    try:
        transcript = loader.load()
        if not transcript:
            print("Transcript is empty. The video may not have captions or the captions are unavailable.")
        else:
            # print(transcript)
            text_splitter= RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=100)
            docs=text_splitter.split_documents(transcript)
            db=FAISS.from_documents(docs,embeddings)
            return db
            
    except Exception as e:
        print(f"An error occurred while loading the transcript: {e}")
    

def get_response_query(db,query,k=4):
    docs=db.similarity_search(query,k=k)
    docs_content=' '.join([d.page_content for d in docs])

    llm=ChatOpenAI(
        model='gpt-4o'
    )
    
    promp_template=PromptTemplate(
        imput_variable=['question','docs'],
        template=""" 
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """
    )
    prompt=promp_template.format(question=query, docs=docs_content)
    # chain=LLMChain(llm=llm,prompt=prompt)
    # response=chain.run(question=query, docs=docs_content)
    response=llm.invoke(prompt)
    response_content=response.content
    response=response_content.replace('\n', " ")
    return response,docs
    
# if __name__=='__main__':
#     db= create_vector_db_from_youtube_urls('https://www.youtube.com/watch?v=rfmpHjmMaXM')
#     print(get_response_query(db,))