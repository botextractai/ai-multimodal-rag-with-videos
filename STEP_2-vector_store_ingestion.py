import lancedb
from mm_rag.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mm_rag.vectorstores.multimodal_lancedb import MultimodalLanceDB
from utils import display_retrieved_results
from utils import load_json_file


# SETUP LANCEDB VECTOR STORE
# ==========================

# declare host file
LANCEDB_HOST_FILE = "./shared_data/.lancedb"
# declare table name
TBL_NAME = "test_tbl"
# initialise vectorstore
db = lancedb.connect(LANCEDB_HOST_FILE)


# PREPARE DATA FOR INGESTION INTO LANCEDB VECTOR STORE
# ====================================================

# load metadata files
vid1_metadata_path = './shared_data/videos/video1/metadatas.json'
vid2_metadata_path = './shared_data/videos/video2/metadatas.json'
vid1_metadata = load_json_file(vid1_metadata_path)
vid2_metadata = load_json_file(vid2_metadata_path)

# collect transcripts and image paths
vid1_trans = [vid['transcript'] for vid in vid1_metadata]
vid1_img_path = [vid['extracted_frame_path'] for vid in vid1_metadata]
vid2_trans = [vid['transcript'] for vid in vid2_metadata]
vid2_img_path = [vid['extracted_frame_path'] for vid in vid2_metadata]

# The existing video caption (frame) texts generated in step 1 are often 
# too short and fragmented. They are thefore not optimal for retrieval. 
# It is therefore required to extend each vido caption text with the 
# video caption texts from "n" neighbouring frames. This results in 
# overlaps, but this # is acceptable, as it is still an improvement. 
# For video1, the value of "n" is set to 7 (frames)
n = 7
updated_vid1_trans = [
 ' '.join(vid1_trans[i-int(n/2) : i+int(n/2)]) if i-int(n/2) >= 0 else
 ' '.join(vid1_trans[0 : i + int(n/2)]) for i in range(len(vid1_trans))
]

# also need to update the updated transcripts in metadata
for i in range(len(updated_vid1_trans)):
    vid1_metadata[i]['transcript'] = updated_vid1_trans[i]


# INGEST DATA INTO LANCEDB VECTOR STORE
# =====================================

# initialise a BridgeTower embedder 
embedder = BridgeTowerEmbeddings()

# passing in mode="append" alows adding more entries to the vector store
# if you want to start with a fresh vector store, you can pass in mode="overwrite" instead 

_ = MultimodalLanceDB.from_text_image_pairs(
    texts=updated_vid1_trans+vid2_trans,
    image_paths=vid1_img_path+vid2_img_path,
    embedding=embedder,
    metadatas=vid1_metadata+vid2_metadata,
    connection=db,
    table_name=TBL_NAME,
    mode="overwrite", 
)
