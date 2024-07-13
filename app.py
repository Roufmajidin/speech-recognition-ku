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
app = Flask(__name__)

# Konfigurasi koneksi ke database MySQL
db_config = {
    'user': 'root',      
    'password': '',   
    'host': 'localhost',    
    'database': 'skripsi_ku',  
}
cache_dir = "models/cache"
processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-base-ar-quran", cache_dir=cache_dir)
model = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-base-ar-quran", cache_dir=cache_dir)
model.config.forced_decoder_ids = None

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def transcribe_audio(audio_data, sampling_rate):
    input_features = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]  # Ambil transkripsi pertama
    return transcription

def convert_m4a_to_mp3(input_path, output_path):
    audio = AudioSegment.from_file(input_path, format="m4a")
    audio.export(output_path, format="mp3")

def segment_phonemes(transcription):
    # Pisahkan transkripsi menjadi fonem-fonem
    # Di sini Anda bisa menggunakan pustaka atau metode lain untuk melakukan segmentasi fonem
    # Contoh sederhana: pisahkan teks dengan spasi
    phonemes = transcription.split()
    return phonemes
def clean_phoneme_list(phoneme_list):
    # Bersihkan elemen dalam daftar dari karakter tambahan
    return [phoneme.strip("[] ,") for phoneme in phoneme_list]

@app.route('/getmateri', methods=['GET'])

def get_materi():
    # Ambil data dari tabel materi
    materi = get_materi_from_db()

    # Kembalikan data sebagai respons JSON
    return jsonify(materi)
@app.route('/get_audio/<path:filename>', methods=['GET'])
def get_audio(filename):
    audio_directory = 'uploads'
    audio_path = os.path.join(audio_directory, filename)
    if os.path.isfile(audio_path):
        return send_from_directory(audio_directory, filename)
    else:
        app.logger.error(f"File not found: {audio_path}")
        return "File not found", 404
    
#TODO Fungsi untuk mengambil data materi berdasarkan device_pengguna
@app.route('/get_materi_by_device_pengguna/<device_pengguna>', methods=['GET'])
def get_materi_by_device_pengguna(device_pengguna):
    try:
        # Koneksi ke database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Query untuk mengambil data pengguna berdasarkan device_pengguna
        query_pengguna = "SELECT * FROM pengguna WHERE device_pengguna = %s"
        cursor.execute(query_pengguna, (device_pengguna,))
        pengguna_data = cursor.fetchall()

        # List untuk menampung hasil akhir
        results = []

        # Loop through pengguna_data to fetch related materi from materii table
        for pengguna in pengguna_data:
            # Query untuk mengambil data materi berdasarkan materi_id dari pengguna
            query_materi = "SELECT * FROM materii WHERE id = %s"
            cursor.execute(query_materi, (pengguna['materi_id'],))
            materi_data = cursor.fetchone()  # Ambil satu baris data materi
            materi_detail = {
                            'audio': materi_data['audio'],
                            'contoh_soal': materi_data['contoh_soal'],
                            'id': materi_data['id'],
                            'jenis_kuis': materi_data['jenis_kuis'],
                            'judul': materi_data['judul'],
                            'kategori': materi_data['kategori'],
                            'is_learn': pengguna['is_learn'],  # Tambahkan is_learn di sini
                            'materi': materi_data['materi']
                        }
            # Buat objek hasil yang mencakup data pengguna dan data materi
            result = {
                'id': pengguna['id'],
                'device_pengguna': pengguna['device_pengguna'],
                'materi_id': pengguna['materi_id'],
                'is_learn': pengguna['is_learn'],
                'materi_detail': materi_detail  # Masukkan detail materi ke dalam hasil
            }

            results.append(result)

        cursor.close()
        conn.close()

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)})

def get_materi_from_db():
    # Koneksi ke database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    # Eksekusi query
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
# validasi pengguna 
def validate_device_pengguna(device_pengguna):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Query untuk memeriksa apakah nama pengguna sudah ada di database
        query = "SELECT COUNT(*) FROM pengguna WHERE device_pengguna = %s"
        cursor.execute(query, (device_pengguna,))
        result = cursor.fetchone()[0]  # Mengambil hasil query, berupa jumlah baris

        cursor.close()
        conn.close()

        # Jika jumlah baris > 0, nama pengguna sudah ada
        return result > 0
    except Exception as e:
        print(f"Error validating device_pengguna: {str(e)}")
        return True  # Secara default, anggap validasi gagal jika terjadi kesalahan

# strore data pengguna 
# Endpoint POST untuk menyimpan data pengguna dan materi ke tabel pengguna dalam satu transaksi
@app.route('/store_pengguna', methods=['POST'])
def store_pengguna():
    data = request.get_json()
    device_pengguna = data.get('device_pengguna')
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    try:
        if validate_device_pengguna(device_pengguna):
            return jsonify({'error': 'Device pengguna already exists'}), 400

       

        conn.start_transaction()

        for item in get_materi_from_db():
            if item['id'] is not None:
                query_materi = "INSERT INTO pengguna (device_pengguna, materi_id, is_learn, persentase) VALUES (%s, %s, %s, %s)"
                values_materi = (device_pengguna, item['id'], False, 0)
                cursor.execute(query_materi, values_materi)

        conn.commit()

        return jsonify({'message': 'Data pengguna and materi berhasil disimpan'})
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()
# Fungsi untuk memperbarui data di tabel pengguna
@app.route('/update_materi_pengguna/<int:id>', methods=['POST'])
def update_materi_pengguna(id):
    try:
        # Ambil parameter query materi_id dari URL
        materi_id = id
        # Ambil data dari body request
        data = request.get_json()
        persentase = data.get('persentase')
        is_learn = data.get('is_learn')

        if persentase is None or is_learn is None:
            return jsonify({'error': 'persentase and is_learn are required'}), 400

        # Koneksi ke database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Query untuk memperbarui data pengguna
        query_update = """
        UPDATE pengguna 
        SET persentase = %s, is_learn = %s 
        WHERE materi_id = %s AND materi_id = %s
        """
        values_update = (persentase, is_learn, id, materi_id)
        cursor.execute(query_update, values_update)

        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({'message': 'Data pengguna berhasil diperbarui'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/', methods=['POST'])
def index():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Simpan file audio yang diunggah secara sementara
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        file.save(temp_audio_file_path)

        # Konversi file M4A ke MP3 jika diperlukan
        mp3_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_output.mp3')
        convert_m4a_to_mp3(temp_audio_file_path, mp3_output_path)

        # Baca file audio
        audio_data, sampling_rate = librosa.load(mp3_output_path, sr=16000, mono=True)
         # Lakukan denoise pada audio
        denoised_audio = reduce_noise(y=audio_data, sr=16000)

        # Simpan file denoise sementara
        temp_denoised_wav_path = os.path.join(tempfile.gettempdir(), 'temp_denoised.wav')
        write(temp_denoised_wav_path, sampling_rate, denoised_audio)

        # Baca file denoise
        denoised_audio_data, _ = librosa.load(temp_denoised_wav_path, sr=16000, mono=True)
    
        # Transkripsi audio
        transcription = transcribe_audio(denoised_audio_data, sampling_rate)

        # Segmentasi fonem
        phonemes = segment_phonemes(transcription)

        # Hapus file audio sementara setelah pemrosesan
        temp_audio_file.close()
        os.remove(temp_audio_file_path)
        print(transcription)
        # Kembalikan fonem-fonem sebagai respons JSON
        return jsonify({'text': transcription})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
