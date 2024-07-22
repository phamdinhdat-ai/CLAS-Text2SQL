import torch
import pyaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer , AutoModelForSeq2SeqLM
import keyboard
import time
from utils import json, loadJsonFile
import sqlite3
from typing import List, Optional, Dict
import pandas as pd 
from text_sim import get_top_k_similar
# from embedd_data import embedd_query
import torchaudio
import gradio as gr 

SPEECH_MODEL = 'whisper-tiny'
MODEL_PATH = "./flan-t5-text2sql-with-schema"
TOKENIZER_PATH = "./flan-t5-text2sql-with-schema"
COLUMNS_JSON_FILE = "./data/columns.json"

# Initialize the processor and model

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(TOKENIZER_PATH)
model.eval()
# define speech model
processor = WhisperProcessor.from_pretrained(SPEECH_MODEL, language="en")
whisper = WhisperForConditionalGeneration.from_pretrained(SPEECH_MODEL)
whisper.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
whisper.to(device)


# PyAudio parameters
CHUNK = 1024  # Number of audio samples per buffer
FORMAT = pyaudio.paInt16  # Format of the audio
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate in Hz
RECORD_SECONDS = 7    # Duration to record before transcribing

#################### INITIAL DATABASE ######################
print("Hi! I am a SQL Query Generator. I can generate SQL queries for you. Please enter the table name for which you want to generate the SQL query.")
table_name = input("Enter the table name : ")

columns = loadJsonFile("../data/columns.json", verbose=False)
table = columns[table_name]
DB_FILEPATH = "../data/database/final_db.db"
print("Load SQL database .....")
conn = sqlite3.connect(DB_FILEPATH)
######################## INFERENCE TIME #########################
# Initialize PyAudio
p = pyaudio.PyAudio()
########################## define function ########################
def prepare_input(question: str, table: List[str]):
    table_prefix = "table:"
    table_name_prefix = "table_name:"
    sample_prefix = ""
    question_prefix = "question:"
    join_table = ",".join(table)
    inputs = f"""
    You are an SQL Query expert who can write SQL queries for the below table.

    {table_prefix} {join_table}

    Answer the following question:
    question : {question}
    """
    input_ids = tokenizer(inputs, max_length=512, return_tensors="pt").input_ids
    return input_ids



def cot_prepare_input(question: str, table: List[str], questions : List[str], example_queries : List[str]):
    table_prefix = "table:"
    table_name_prefix = "table_name:"
    sample_prefix = ""
    question_prefix = "question:"
    join_table = ",".join(table)
    inputs = f"""
    You are an SQL Query expert who can write SQL queries for the below table.
    {table_prefix} {join_table}
    For the below questions, you are given the example queries. You need to write the SQL query for the last question.
    """
    for question_no, s_question in enumerate(questions):
        inputs += f"""
        {s_question}
        {example_queries[question_no]}
        """

    inputs += f"""
    Only answer the following question:
    {question},
    """
    input_ids = tokenizer(inputs, max_length=512, return_tensors="pt").input_ids
    return input_ids


def inference(question: str, table: List[str]) -> str:
    input_data = prepare_input(question=question, table=table)
    input_data = input_data.to(model.device)
    outputs = model.generate(inputs=input_data, num_beams=10, top_k=10, max_length=1024)
    result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
    return result



def cot_inference(question: str, table: List[str], questions : List[str], example_queries : List[str]) -> str:
    input_data = cot_prepare_input(question=question, table=table, questions = questions, example_queries = example_queries)
    input_data = input_data.to(model.device)
    outputs = model.generate(inputs=input_data, num_beams=10, top_k=10, max_length=1024)
    result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
    return result




# DB_FILEPATH = "../data/database/final_db.db"
def text2sql(question, conn, k = 5, table_name = "employee"):
    
    retr_df = pd.read_csv("../data/example_queries/retr_set/final_" +  table_name  + ".csv", delimiter="|")

    sample_questions = []
    sample_queries = []

    print('\n')
    
    # get reference question and sql queries
    for index, row in retr_df.iterrows():
        s_question = row[' Question']
        s_queries = row['SQL Query']
        sample_questions.append(s_question)
        sample_queries.append(s_queries)

    print("Retrieving top " + str(k)  + " similar queries from the dataset to perform Chain of Thought (CoT) Prompting ...")
    top_k_indices = get_top_k_similar(question, sample_questions, k= k)

    sample_questions = [sample_questions[i] for i in top_k_indices]
    sample_queries = [sample_queries[i] for i in top_k_indices]

    for question_no, s_question in enumerate(sample_questions):
        print("Question : ", sample_questions[question_no])
        print("SQL Query :", sample_queries[question_no])

    print("\n")
    print(" ========= Zero-Shot Test SQL ========= ")
    gen_zero_shot = inference(question, table)
    gen_zero_shot = gen_zero_shot.replace(" table", " " + table_name)
    
    print("Generated Query using Zero-Shot \nPrompting = ", gen_zero_shot)
    try:
        zero_shot_result = conn.exercute(gen_zero_shot).fetchall()
        print(conn.execute(gen_zero_shot).fetchall())
        print("Zero-Shot Query Works!")
    except:
        zero_shot_result = "Error in Zero-Shot SQL Query"
        print("Error in Zero-Shot SQL Query")
    
    print(" ========= CoT Test SQL ========= ")
    
    
    
    gen_cot = cot_inference(question, table, sample_questions, sample_queries)
    gen_cot = gen_cot.replace(" table", " " + table_name)
    
    print("Generated Query using Chain of Thought (CoT) Prompting = ", gen_cot)
    try:
        cot_results = conn.execute(gen_cot).fetchall()
        print(conn.execute(gen_cot).fetchall())
        print("CoT Query Works!")
    except:
        cot_results = "Error in CoT SQL Query!"
        print("Error in CoT SQL Query!")

    return zero_shot_result, cot_results

########################## define function ########################
def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate

def transcribe_audio(model, processor, audio_path):
    waveform, sample_rate = load_audio(audio_path)
    # Thay đổi sample rate của âm thanh về 16000 Hz (nếu cần)
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)
    # Xử lý âm thanh để tạo đầu vào cho mô hình
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
    inputs = inputs.to(device)
    # Thực hiện inference
    predicted_ids = model.generate(inputs)
    # Chuyển đổi các ID dự đoán thành văn bản
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription    


def process_inference(audio_path, table_name):
    
    columns = loadJsonFile("../data/columns.json", verbose=False)
    table = columns[table_name]
    transcription = transcribe_audio(model=whisper, processor=processor, audio_path=audio_path)
    print("transcription: ", transcription)
    zero_shot_result, cot_results = text2sql(transcription, conn=conn, k=5, table_name=table_name)

    return transcription, zero_shot_result, cot_results


if __name__ == "__main__":
    iface = gr.Interface(
    fn=process_inference,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Textbox(type="text", label="Enter name of Database")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Zero Shot result"),
        gr.Textbox(label="Cot SQL query"),
        
    ],
    title="The comprehensive Library Asssistant System",
    description="This is a demo of CLAS which created by Dat and Tu",
    
)

# Launch the interface
iface.launch(debug=True, share=True)