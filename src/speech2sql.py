### library for speech recognitions
import sounddevice as sd 
import numpy as np 
import torch 
from transformers import WhisperProcessor, WhisperForConditionalGeneration

### Libray for Text2sql

import os 
from utils import dumpJsonFile, loadJsonFile
import sqlite3
import pandas as pd
from text_sim import *
from tqdm import tqdm
import time

import warnings
warnings.filterwarnings('ignore')


from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#################################  DEFINE ARGUMENTS #############################
# Model taken from Huggingface Github Repo
SPEECH_MODEL = 'whisper-tiny'
MODEL_PATH = "./flan-t5-text2sql-with-schema"
TOKENIZER_PATH = "./flan-t5-text2sql-with-schema"
COLUMNS_JSON_FILE = "./data/columns.json"

# Thiết lập các tham số cho thu âm
RATE = 16000
BUFFER_DURATION = 10 # Thời gian của mỗi đoạn buffer thu âm (giây)
BUFFER_SIZE = int(RATE * BUFFER_DURATION)  # Kích thước của buffer

############################## LOAD LANGUAGE AND SPEECH MODEL ###############################
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(TOKENIZER_PATH)

processor = WhisperProcessor.from_pretrained(SPEECH_MODEL, language="en")
whisper = WhisperForConditionalGeneration.from_pretrained(SPEECH_MODEL)
# Biến lưu buffer âm thanh
buffer = np.zeros(BUFFER_SIZE, dtype='float32')
current_pos = 0

# Biến lưu trữ chuỗi mới
new_transcript = ""
listening_for_command = False



############################## DEFINE HELPER LANGUAGE FUNCTION #################################
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
    top_k_indices = get_top_k_similar(question, sample_questions, k = k)

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
        print(conn.execute(gen_zero_shot).fetchall())
        print("Zero-Shot Query Works!")
    except:
        print("Error in Zero-Shot SQL Query")
    
    print(" ========= CoT Test SQL ========= ")
    
    
    
    gen_cot = cot_inference(question, table, sample_questions, sample_queries)
    gen_cot = gen_cot.replace(" table", " " + table_name)
    
    print("Generated Query using Chain of Thought (CoT) Prompting = ", gen_cot)
    try:
        print(conn.execute(gen_cot).fetchall())
        print("CoT Query Works!")
    except:
        print("Error in CoT SQL Query!")


############################## DEFINE HELPER SPEECH FUNCTION ##################################
# Hàm để chuyển đổi âm thanh thành văn bản
def transcribe_audio(whisper, processor, waveform):
    inputs = processor(waveform, sampling_rate=RATE, return_tensors="pt").input_features
    predicted_ids = whisper.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

#TODO 
def post_processing_audio():
    pass

# Hàm callback để thu âm
def audio_callback(indata, frames, time, status):
    global buffer, current_pos, new_transcript, listening_for_command, table, table_name, conn 
    if status:
        print(status)

    # Thêm dữ liệu mới vào buffer
    if current_pos + frames < BUFFER_SIZE:
        buffer[current_pos:current_pos + frames] = indata[:, 0]
        current_pos += frames
    else:
        
        # Khi buffer đầy, xử lý buffer và đặt lại vị trí hiện tại
        buffer[current_pos:] = indata[:BUFFER_SIZE - current_pos, 0]
        waveform = buffer.copy()
        current_pos = frames - (BUFFER_SIZE - current_pos)
        buffer[:current_pos] = indata[BUFFER_SIZE - current_pos:, 0]
        
        # Xử lý và dịch buffer thành văn bản
        transcription = transcribe_audio(whisper, processor, waveform)
        print("Your Question:", transcription)

        text2sql(question=transcription, conn=conn, k=2, table_name=table_name)
        print("Next question !")
        sd.sleep(1000)
        
        # if "hey zara" in transcription.lower():
        #     listening_for_command = True
        #     print("Bắt đầu ghi lại các từ sau 'Hey Zara'")

        # if listening_for_command:
        #     if "stop zara" in transcription.lower():
        #         print("Ngừng ghi lại khi nghe thấy 'Stop Zara'")
        #         listening_for_command = False
        #     else:
        #         new_transcript += " " + transcription
        #         print("new_transcript:", new_transcript)






if __name__ == '__main__':
    
    print("Hi! I am a SQL Query Generator. I can generate SQL queries for you. Please enter the table name for which you want to generate the SQL query.")
    table_name = input("Enter the table name : ")

    columns = loadJsonFile("../data/columns.json", verbose=False)
    table = columns[table_name]
    DB_FILEPATH = "../data/database/final_db.db"
    print("Load SQL database .....")
    conn = sqlite3.connect(DB_FILEPATH)
    for i in tqdm(range(2)):
        time.sleep(1)
        
    print("\n")
    
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=RATE, blocksize=BUFFER_SIZE):
        print("Đang thu âm và dịch trực tiếp... Nhấn Ctrl+C để dừng.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Dừng thu âm.")
            print("Final new_transcript:", new_transcript)
    