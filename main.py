from openai import OpenAI
from dotenv import load_dotenv
import os

def call_openai_api():
    """
    Load the API key from environment variables and initialize the OpenAI client.
    Returns:
        OpenAI client object.
    """
    # Load environment variables from .env
    load_dotenv()

    # Retrieve the API key from the environment variables
    api_key = os.getenv("OPENAI_API_KEY")

    # Make sure the API key is valid
    if not api_key:
        raise ValueError("API key not found. Make sure to store it in a .env file.")

    # Set the API key for OpenAI
    client = OpenAI(api_key=api_key)
    return client

def transcribe_audio(client, input_audio_fn):
    """
    Transcribe the given audio file using the OpenAI client.
    Args:
        client: OpenAI client object.
        input_audio_fn: Path to the input audio file.
    Returns:
        Dictionary of transcriptions in various formats.
    """
    formats = ["json", "text", "srt", "vtt"]
    transcripts = {}

    for fmt in formats:
        with open(input_audio_fn, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format=fmt
            )
            transcripts[fmt] = response

    # Handle verbose_json separately for word and segment timestamps
    with open(input_audio_fn, "rb") as audio_file:
        response_word = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
        transcripts["verbose_json_word"] = response_word

    with open(input_audio_fn, "rb") as audio_file:
        response_segment = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
        transcripts["verbose_json_segment"] = response_segment

    return transcripts

def save_text_to_file(text, output_filename):
    """
    Save the given text to a file.
    Args:
        text: Text to save.
        output_filename: Path to the output file.
    """
    # Write the text to a text file
    with open(output_filename, "w") as text_file:
        text_file.write(text)

def main():
    audio_file_path = "/Users/duc/Desktop/Marketing Insider/Đổi tên kênh/20240718_C0925.mp3"
    client = call_openai_api()

    # Transcribe the audio file
    transcripts = transcribe_audio(client, audio_file_path)

    # Save each transcription format to a file
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    for format, response in transcripts.items():
        if "verbose_json" in format:
            content = response.json()
            output_filename = f"{base_name}_{format}.json"
        elif format in ["json"]:
            content = response.json()
            output_filename = f"{base_name}.{format}"
        else:
            content = str(response)
            output_filename = f"{base_name}.{format}"

        save_text_to_file(content, output_filename)
        print(f"Transcription saved to {output_filename}")

if __name__ == "__main__":
    main()
