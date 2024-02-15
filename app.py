import gradio as gr

from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from pymilvus import Collection, connections
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "LangChainCollection")

MILVUS_HOST = os.environ.get("MILVUS_HOST", "")
MILVUS_PORT = "19530"

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "hkunlp/instructor-large")
EMBEDDING_LOADER = os.environ.get("EMBEDDING_LOADER", "HuggingFaceInstructEmbeddings")
EMBEDDING_LIST = ["HuggingFaceInstructEmbeddings", "HuggingFaceEmbeddings"]

# return top-k text chunks from vector store
TOP_K_DEFAULT = 15
TOP_K_MAX = 30
SCORE_DEFAULT = 0.33

BUTTON_MIN_WIDTH = 100

global g_emb
g_emb = None
global g_col
g_col = None

def init_emb(emb_name, emb_loader, db_col_textbox):
    
    global g_emb
    global g_col

    g_emb = eval(emb_loader)(model_name=emb_name)
    
    connections.connect( 
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    
    g_col = Collection(db_col_textbox)
    
    g_col.load()
    
    return (str(g_emb), str(g_col))


def get_emb():
    return g_emb

def get_col():
    return g_col


def remove_duplicates(documents, score_min):
    seen_content = set()
    unique_documents = []
    for (doc, score) in documents:
        if (doc.page_content not in seen_content) and (score >= score_min):
            seen_content.add(doc.page_content)
            unique_documents.append(doc)
    return unique_documents


def get_data(query, top_k, score, db_col, db_index):
    if not query:
        return "Please init db in configuration"

    embed_query = g_emb.embed_query(query)
    
    search_params = {"metric_type": "L2",
                     "params": {"nprobe": 1},
                     "offset": 0}
    
    
    results = g_col.search(
        data=[embed_query], 
        anns_field="vector",
        param=search_params,
        limit=top_k, 
        expr=None,
        output_fields=['source', 'text'],
        consistency_level="Strong"
    )
    
    jsons = json.dumps([{'source': hit.entity.get('source'),
                        'text': hit.entity.get('text')}
                       for hit in results[0]],
                      indent=0)

    return jsons

with gr.Blocks(
    title = "3GPP Database",
    theme = "Base",
    css = """.bigbox {
    min-height:250px;
}
""") as demo:
    with gr.Tab("Matching"):
        with gr.Accordion("Vector similarity"):
            with gr.Row():
                with gr.Column():
                    top_k = gr.Slider(1,
                                      TOP_K_MAX,
                                      value=TOP_K_DEFAULT,
                                      step=1,
                                      label="Vector similarity top_k",
                                      interactive=True)
                with gr.Column():
                    score = gr.Slider(0.01,
                                      0.99,
                                      value=SCORE_DEFAULT,
                                      step=0.01,
                                      label="Vector similarity score",
                                      interactive=True)

        with gr.Row():
            with gr.Column(scale=10):
                input_box = gr.Textbox(label = "Input", placeholder="What are you looking for?")
            with gr.Column(scale=1, min_width=BUTTON_MIN_WIDTH):
                btn_run = gr.Button("Run", variant="primary")

        output_box = gr.JSON(label = "Output")


    with gr.Tab("Configuration"):
        with gr.Row():
            btn_init = gr.Button("Init")
        
        load_emb = gr.Textbox(get_emb, label = 'Embedding Client', show_label=True)
        load_col = gr.Textbox(get_col, label = 'Milvus Collection', show_label=True)
        
        with gr.Accordion("Embedding"):
                
            with gr.Row():
                with gr.Column():
                    emb_textbox = gr.Textbox(
                        label = "Embedding Model",
                        # show_label = False,
                        value = EMBEDDING_MODEL,
                        placeholder = "Paste Your Embedding Model Repo on HuggingFace",
                        lines=1,
                        interactive=True,
                        type='email')

                with gr.Column():
                    emb_dropdown = gr.Dropdown(
                        EMBEDDING_LIST,
                        value=EMBEDDING_LOADER,
                        multiselect=False,
                        interactive=True,
                        label="Embedding Loader")

        with gr.Accordion("Milvus Database"):
            with gr.Row():
                db_col_textbox = gr.Textbox(
                    label = "Milvus Collection",
                    # show_label = False,
                    value = MILVUS_COLLECTION,
                    placeholder = "Paste Your Milvus Collection (xx-xx-xx) and Hit ENTER",
                    lines=1,
                    interactive=True,
                    type='email')
                db_index_textbox = gr.Textbox(
                    label = "Milvus Host",
                    # show_label = False,
                    value = MILVUS_HOST,
                    placeholder = "Paste Your Milvus Index (xxxx) and Hit ENTER",
                    lines=1,
                    interactive=True,
                    type='password')

    btn_init.click(fn=init_emb,
                   inputs=[emb_textbox, emb_dropdown, db_col_textbox],
                   outputs=[load_emb, load_col])
    btn_run.click(fn=get_data,
                  inputs=[input_box, top_k, score, db_col_textbox, db_index_textbox],
                  outputs=[output_box])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0",
                server_port=7860)

