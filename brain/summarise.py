from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import json

with open('searchResults/scraped_data.json', 'r') as f:
    results = json.load(f)

def generate_summary(results, query, index):
    

    llm = ChatOllama(
        model="llama3.1:8b-instruct-q4_K_M",
        temperature=0,
       # other params...
    )
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Your task is to accurately and concisely explain and elaborate on the content of the top search results related to the query: '{query}'. \
            You will receive input as scraped data from relevant websites. \
            Your output should be simple, clear, and easy to understand, even for someone with no prior knowledge of the topic. \
            Focus on providing the most relevant and informative explanation, avoiding any technical jargon. \
            Please exclude any promotional or irrelevant content. \
            The primary objective is to deliver high-quality, accessible information for the query: '{query}'.\
            Here is the scraped data from the top website content results: {results}\
            There are links in the scraped data, retain them in the output.\
            If results say somehting related to denial of access, please ignore that and proceed with the next result.\
            Current index: {index}\
            RULE - OUTPUT SHOULD BE A MARKDOWN FORMATTED TEXT. RETAIN THE LINKS IN THE OUTPUT.",
        ),
        ("human", '${query}'),
    ]
    )
    chain = prompt | llm
    ai_msg = chain.invoke(
                {
                    "query":query, 
                    "results":results, 
                    "index":index
                }
    )


    return ai_msg.content

def main():
    for i in results['results']:
        title = i['title']
        link = i['link']
        text = i['text']
        # summary = generate_summary(text)
        print(f"Title: {title}\nLink: {link}\nSummary: {text}\n\n")

if __name__ == '__main__':
    main()
