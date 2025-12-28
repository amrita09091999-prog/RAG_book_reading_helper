import json
import os 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-14B-Instruct",
            task="text-generation",
            temperature = 0.2
            )
judge_model = ChatHuggingFace(llm=llm)

# import rag_response
with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/rag_response.json",'r') as f:
    rag_response = json.load(f)

query = rag_response['query']
context = rag_response['retrieved_context']
answer = rag_response['response']

# answer relevance 

with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/prompts/evaluation_prompts/answer_relevance.txt",'r') as f:
    answer_relevance_template = f.read()

answer_relevance_prompt_template = PromptTemplate(
    template = answer_relevance_template,
    input_variables = ['question','rag_response']
)
answer_relevance_prompt = answer_relevance_prompt_template.format(
    question = query,
    rag_response = answer
)

# groundness

with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/prompts/evaluation_prompts/groundness.txt",'r') as f:
    groundness_template = f.read()

groundness_prompt_template = PromptTemplate(
    template = groundness_template,
    input_variables = ['rag_response','context']
)
groundness_prompt = groundness_prompt_template.format(
    rag_response = answer,
    context = context
)

# retrieval relevence

with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/prompts/evaluation_prompts/retrieval_relevance.txt",'r') as f:
    retrieval_relevence_template = f.read()

retrieval_relevence_prompt_template = PromptTemplate(
    template = retrieval_relevence_template,
    input_variables = ['query','context']
)
retrieval_relevence_prompt = retrieval_relevence_prompt_template.format(
    query = query,
    context = context
)

with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/rag_response.json",'r') as f:
    rag_response = json.load(f)


print("answer relevence")
answer_relevance_response = judge_model.invoke(answer_relevance_prompt)
print("\n")

print('groundness')
groundness_response = judge_model.invoke(groundness_prompt)
print("\n")
print('retrieval relevence')
retrieval_relevence_response = judge_model.invoke(retrieval_relevence_prompt)

text = f"""
User Query - {query}\n
AI Response - {answer}\n
Answer Relevance - \n
 {answer_relevance_response.content}\n
 Groundness - \n
 {groundness_response.content}\n
 Retrieval Relevance - \n
 {retrieval_relevence_response.content}"""

os.makedirs("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/evaluation",exist_ok = True)

with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/evaluation/evaluation.txt",'w') as f:
    f.write(text)
# evaluation = {
#     "User Query":query,
#     "AI Response":answer,
#     "Answer Relevance":{
#         'Score':json.loads(answer_relevance_response.content)['score'],
#         'Explanation':json.loads(answer_relevance_response.content)['Explanation']
#     },
#     "Groundness":{
#         'Score':json.loads(groundness_response.content)['score'],
#         'Explanation':json.loads(groundness_response.content)['Explanation']
#     },
#     "Retrieval Relevance":{
#         'Score':json.loads(retrieval_relevence_response.content)['score'],
#         'Explanation':json.loads(retrieval_relevence_response.content)['Explanation']
#     }
# }
# with open("/Users/amritamandal/Desktop/Python/Projects/Novel_Reading_Assistant/RAG_book_reading_helper-1/RAG_clean/rag_response_bundle/evaluation/evaluation.json",'w') as f:
#     json.dump(evaluation)