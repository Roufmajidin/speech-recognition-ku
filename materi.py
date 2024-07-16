from flask import Flask, request, jsonify, send_from_directory
from pydub import AudioSegment
import os
import tempfile
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import mysql.connector
import json
from scipy.io.wavfile import write
from noisereduce import reduce_noise

db_config = {
    'user': 'root',      
    'password': '',   
    'host': 'localhost',    
    'database': 'skripsi_ku',  
}

def clean_phoneme_list(phoneme_list):
    # Bersihkan elemen dalam daftar dari karakter tambahan
    return [phoneme.strip("[] ,") for phoneme in phoneme_list]

def clean_phoneme_list(phoneme_list):
    # Implement your phoneme list cleaning logic here
    return phoneme_list

def get_materi_from_db():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM materii")
    materi = cursor.fetchall()

    for item in materi:
        if 'contoh_soal' in item and item['contoh_soal']:
            try:
                phoneme_list = json.loads(item['contoh_soal'])
                item['contoh_soal'] = clean_phoneme_list(phoneme_list)
            except json.JSONDecodeError:
                item['contoh_soal'] = clean_phoneme_list(item['contoh_soal'].split())
        if 'audio' in item and item['audio']:
            item['audio_url'] = f"/get_audio/{item['audio']}"

    cursor.close()
    conn.close()
    return materi

def fetch_materi_by_id(materi_id):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM materii WHERE id = %s"
    cursor.execute(query, (materi_id,))
    materi = cursor.fetchone()

    if materi:
        if 'contoh_soal' in materi and materi['contoh_soal']:
            try:
                phoneme_list = json.loads(materi['contoh_soal'])
                materi['contoh_soal'] = clean_phoneme_list(phoneme_list)
            except json.JSONDecodeError:
                materi['contoh_soal'] = clean_phoneme_list(materi['contoh_soal'].split())
        if 'audio' in materi and materi['audio']:
            materi['audio_url'] = f"/get_audio/{materi['audio']}"

    cursor.close()
    conn.close()
    return materi

def fetch_materi_by_device_pengguna(device_pengguna):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    query_pengguna = "SELECT * FROM pengguna WHERE device_pengguna = %s"
    cursor.execute(query_pengguna, (device_pengguna,))
    pengguna_data = cursor.fetchall()

    results = []
    for pengguna in pengguna_data:
        query_materi = "SELECT * FROM materii WHERE id = %s"
        cursor.execute(query_materi, (pengguna['materi_id'],))
        materi_data = cursor.fetchone()

        if materi_data:
            materi_detail = {
                'audio': materi_data['audio'],
                'contoh_soal': materi_data['contoh_soal'],
                'id': materi_data['id'],
                'jenis_kuis': materi_data['jenis_kuis'],
                'judul': materi_data['judul'],
                'kategori': materi_data['kategori'],
                'is_learn': pengguna['is_learn'],
                'materi': materi_data['materi']
            }
            result = {
                'id': pengguna['id'],
                'device_pengguna': pengguna['device_pengguna'],
                'materi_id': pengguna['materi_id'],
                'is_learn': pengguna['is_learn'],
                'materi_detail': materi_detail
            }
            results.append(result)

    cursor.close()
    conn.close()
    return results

    # Koneksi ke database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    # Eksekusi query
    query = "SELECT * FROM materii WHERE id = %s"
    cursor.execute(query, (materi_id,))
    materi = cursor.fetchone()

    if materi:
        if 'contoh_soal' in materi and materi['contoh_soal']:
            try:
                phoneme_list = json.loads(materi['contoh_soal'])
                materi['contoh_soal'] = clean_phoneme_list(phoneme_list)
            except json.JSONDecodeError:
                materi['contoh_soal'] = clean_phoneme_list(materi['contoh_soal'].split())
        if 'audio' in materi and materi['audio']:
            materi['audio_url'] = f"/get_audio/{materi['audio']}"

    cursor.close()
    conn.close()

    return materi

def get_current_materi(materi_id):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM materii WHERE id = %s", (materi_id,))
    current_data = cursor.fetchone()

    if current_data:
        keys = ['id', 'jenis_kuis', 'kategori', 'judul', 'materi', 'audio', 'contoh_soal']
        return dict(zip(keys, current_data))

    cursor.close()
    conn.close()
    return {}


def update_materi(data):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    fields = []
    values = []

    # Update only the fields that are not None
    if 'audio' in data and data['audio'] is not None:
        fields.append("audio = %s")
        values.append(data['audio'])
    if 'contoh_soal' in data and data['contoh_soal'] is not None:
        fields.append("contoh_soal = %s")
        values.append(json.dumps(data['contoh_soal']))
    if 'jenis_kuis' in data and data['jenis_kuis'] is not None:
        fields.append("jenis_kuis = %s")
        values.append(data['jenis_kuis'])
    if 'judul' in data and data['judul'] is not None:
        fields.append("judul = %s")
        values.append(data['judul'])
    if 'kategori' in data and data['kategori'] is not None:
        fields.append("kategori = %s")
        values.append(data['kategori'])
    if 'materi' in data and data['materi'] is not None:
        fields.append("materi = %s")
        values.append(data['materi'])

    if not fields:
        return False

    values.append(data['id'])
    query = f"UPDATE materii SET {', '.join(fields)} WHERE id = %s"

    cursor.execute(query, values)
    conn.commit()
    success = cursor.rowcount > 0
    cursor.close()
    conn.close()

    return success

def insert_materi(data):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    query = """
        INSERT INTO materii (jenis_kuis, kategori, judul, materi, audio, contoh_soal)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (
        data['jenis_kuis'],
        data['kategori'],
        data['judul'],
        data['materi'],
        data['audio'],
        json.dumps(data['contoh_soal'])
    )

    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    conn.close()

    return cursor.rowcount > 0

def delete_materi(materi_id):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    query = "DELETE FROM materii WHERE id = %s"
    cursor.execute(query, (materi_id,))
    conn.commit()
    cursor.close()
    conn.close()

    return cursor.rowcount > 0