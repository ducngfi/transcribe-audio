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
        Transcription text.
    """
  # Open and read the audio file
  with open(input_audio_fn, "rb") as audio_file:
    # Use the correct method to transcribe the audio
    response = client.audio.transcriptions.create(
      model="whisper-1", 
      file=audio_file,
    )
    #print(response)
    transcript = response.text

  # returns the transcripted text
  return transcript

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
  transcription_file = "transcription.txt"

  client = call_openai_api()

  # Transcribe the audio file
  transcript = transcribe_audio(client, audio_file_path)

  # Print the transcription
  print("Transcription:")
  print(transcript)

  # Save the transcription to a text file
  save_text_to_file(transcript, transcription_file)
  print(f"Transcription saved to {transcription_file}")

if __name__ == "__main__":
    main()
