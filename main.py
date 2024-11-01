import streamlit as st 
import langchain_helper as lch
import textwrap

st.title("Youtube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url=st.sidebar.text_area(
            label='What is  Youtube video URl?',
            max_chars=50
        )

        
        
        
        
query=st.text_area(
    label='Ask me qustion about the video?',
    key='query'
            
)
    
if query and youtube_url:
    db=lch.create_vector_db_from_youtube_urls(youtube_url)
    response,docs=lch.get_response_query(db,query)
    st.subheader("Answear:")
    st.text(textwrap.fill(response,width=80))