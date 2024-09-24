from langchain_core.runnables import (
    RunnableParallel, 
    RunnablePassthrough, 
    RunnableLambda
)
from mm_rag.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mm_rag.MLM.client import PredictionGuardClient
from mm_rag.MLM.lvlm import LVLM
from mm_rag.vectorstores.multimodal_lancedb import MultimodalLanceDB
from PIL import Image
from utils import load_json_file


# PREPROCESSING
# =============

# declare host file
LANCEDB_HOST_FILE = "./shared_data/.lancedb"

# declare table name
TBL_NAME = "test_tbl"


# RETRIEVAL MODULE
# ================

# initialise a BridgeTower embedder 
embedder = BridgeTowerEmbeddings()

# create a LanceDB vector store 
vectorstore = MultimodalLanceDB(
    uri=LANCEDB_HOST_FILE, 
    embedding=embedder, 
    table_name=TBL_NAME
)

# create a retriever for the vector store
# with search_type="similarity" and search_kwargs={"k": 1} 
retriever_module = vectorstore.as_retriever(
    search_type='similarity', 
    search_kwargs={"k": 1}
)

# invoke the retrieval for a query
query = "What do the astronauts feel about their work?"
retrieved_video_segments = retriever_module.invoke(query)
# get the first retrieved video segment
retrieved_video_segment = retrieved_video_segments[0]

# get all metadata of the retrieved video segment
retrieved_metadata = retrieved_video_segment.metadata['metadata']

# get the extracted frame
frame_path = retrieved_metadata['extracted_frame_path']
# get the corresponding transcript
transcript = retrieved_metadata['transcript']
# get the path to video where the frame was extracted
video_path = retrieved_metadata['video_path']
# get the time stamp when the frame was extracted
timestamp = retrieved_metadata['mid_time_ms']

# display
print(f"Transcript:\n{transcript}\n")
print(f"Path to extracted frame: {frame_path}")
print(f"Path to video: {video_path}")
print(f"Timestamp in ms when the frame was extracted: {timestamp}")
print("Retrieved Frame (image shown separately):")
Image.open(frame_path).show()


# LVLM INFERENCE MODULE
# =====================

# initialise a client as PredictionGuardClient
client = PredictionGuardClient()
# initialise LVLM with the given client
lvlm_inference_module = LVLM(client=client)

# This new query is the augmentation of the previous query
# with the transcript retrieved above 
augmented_query_template = (
    "The transcript associated with the image is '{transcript}'. "
    "{previous_query}"
)
augmented_query = augmented_query_template.format(
    transcript=transcript,
    previous_query=query,
)
print(f"Augmented query is:\n{augmented_query}")

# use the augmented query and the retrieved path-to-image
# as the input to LVLM inference module
input = {'prompt':augmented_query, 'image': frame_path}
response = lvlm_inference_module.invoke(input)

# display the response
print('LVLM Response:')
print(response)


# PROMPT PROCESSING MODULE
# ========================

def prompt_processing(input):
    # get the retrieved results and user's query
    retrieved_results = input['retrieved_results']
    user_query = input['user_query']
    
    # get the first retrieved result by default
    retrieved_result = retrieved_results[0]
    prompt_template = (
      "The transcript associated with the image is '{transcript}'. "
      "{user_query}"
    )
    
    # get all metadata of the retrieved video segment
    retrieved_metadata = retrieved_result.metadata['metadata']

    # get the corresponding transcript
    transcript = retrieved_metadata['transcript']
    # get the extracted frame
    frame_path = retrieved_metadata['extracted_frame_path']
    
    return {
        'prompt': prompt_template.format(
            transcript=transcript, 
            user_query=user_query
        ),
        'image' : frame_path
    }
    
# initialise prompt processing module 
# as a Langchain RunnableLambda of function prompt_processing
prompt_processing_module = RunnableLambda(prompt_processing)

# use the user query and the retrieved results above
input_to_lvlm = prompt_processing_module.invoke(
    {
        'retrieved_results': retrieved_video_segments, 
        'user_query': query
    })

# display output of prompt processing module 
# which is the input to LVLM Inference module
print(input_to_lvlm)


# MULTIMODAL RAG
# ==============

# combine all the modules into a chain 
# to create Multimodal RAG system
mm_rag_chain = (
    RunnableParallel({
        "retrieved_results": retriever_module , 
        "user_query": RunnablePassthrough()
    }) 
    | prompt_processing_module
    | lvlm_inference_module
)

# invoke the Multimodal RAG system with a query
query1 = "What do the astronauts feel about their work?"
final_text_response1 = mm_rag_chain.invoke(query1)
# display
print(f"USER Query: {query1}")
print(f"MM-RAG Response: {final_text_response1}")

# let's try another query
query2 = "What is the name of one of the astronauts?"
final_text_response2 = mm_rag_chain.invoke(query2)
# display
print(f"USER Query: {query2}")
print(f"MM-RAG Response: {final_text_response2}")

# the output of this new chain will be a dictionary
mm_rag_chain_with_retrieved_image = (
    RunnableParallel({
        "retrieved_results": retriever_module , 
        "user_query": RunnablePassthrough()
    }) 
    | prompt_processing_module
    | RunnableParallel({
        'final_text_output': lvlm_inference_module, 
        'input_to_lvlm' : RunnablePassthrough()
    })
)

# let's try again with query2
response3 = mm_rag_chain_with_retrieved_image.invoke(query2)
# display
print("Type of output of mm_rag_chain_with_retrieved_image is:")
print(type(response3))
print(f"Keys of the dict are {response3.keys()}")

# extract final text response and path to extracted frame
final_text_response3 = response3['final_text_output']
path_to_extracted_frame = response3['input_to_lvlm']['image']

# display
print(f"USER Query: {query2}")
print(f"MM-RAG Response: {final_text_response3}")
print("Retrieved frame (image shown separately):")
Image.open(path_to_extracted_frame).show()

# let's try again with another query
query4 = "an astronaut's spacewalk"
response4 = mm_rag_chain_with_retrieved_image.invoke(query4)
# extract results
final_text_response4 = response4['final_text_output']
path_to_extracted_frame4 = response4['input_to_lvlm']['image']
# display
print(f"USER Query: {query4}")
print()
print(f"MM-RAG Response: {final_text_response4}")
print()
print("Retrieved frame (image shown separately):")
Image.open(path_to_extracted_frame4).show()

# get an astronaut's spacewalk with the earth view behind
query5 = (
    "An astronaut's spacewalk with "
    "an amazing view of the earth from space behind"
)
response5 = mm_rag_chain_with_retrieved_image.invoke(query5)
# extract results
final_text_response5 = response5['final_text_output']
path_to_extracted_frame5 = response5['input_to_lvlm']['image']
# display
print(f"USER Query: {query5}")
print()
print(f"MM-RAG Response: {final_text_response5}")
print()
print("Retrieved Frame (image shown separately):")
Image.open(path_to_extracted_frame5).show()
