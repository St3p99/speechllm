import argparse
import logging
import os
import torch
import torchaudio
from transformers import TextStreamer

from speechllm.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_AUDIO_TOKEN_IDX
from speechllm.conversation import conv_templates, default_conversation
from speechllm.model import SpeechLLM, SpeechLLMConfig
from speechllm.dataset.utils import tokenizer_mm_token


def load_audio_into_tensor(audio_path, target_sr=None, device="cpu"):
    """Load audio file and convert to tensor, and move to the specified device."""
    if "'" in audio_path:
        audio_path = audio_path.replace("'", "")
    audio, orig_sr = torchaudio.load(
        audio_path, normalize=True, channels_first=True
    )
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze()
    
    # Resample if target_sr is specified and different
    if target_sr is not None and orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        audio = resampler(audio)
        orig_sr = target_sr  # update sr to target_sr

    # Move audio to the specified device
    audio = audio.to(device)
    return audio, orig_sr


def load_speechllm_model(model_path, device="cuda", torch_dtype=None):
    """Simple model loader for SpeechLLM."""
    # Load config
    config = SpeechLLMConfig.from_pretrained(model_path)
    
    # Create model
    model = SpeechLLM(config)
    
    # Load state dict
    if os.path.isdir(model_path):
        # Look for pytorch_model.bin or model.safetensors
        state_dict_path = None
        if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        elif os.path.exists(os.path.join(model_path, "model.safetensors")):
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(model_path, "model.safetensors"))
            model.load_state_dict(state_dict, strict=False)
        
        if state_dict_path:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
    
    # Move to device and set dtype
    model = model.to(device)
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    
    model.eval()
    return model


def torch_dtype_from_str(dtype_str):
    """Convert string to torch dtype."""
    if dtype_str is None:
        return None
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
    }
    return dtype_map.get(dtype_str, torch.float32)


def main(args):
    """Main CLI function for ASR with SpeechLLM."""
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_speechllm_model(
        args.model_path, 
        device=args.device,
        torch_dtype=args.torch_dtype
    )
    
    # Set up conversation
    conv_mode = getattr(model.config, 'conversation_version', 'llama_3_1')
    if conv_mode not in conv_templates:
        conv_mode = 'llama_3_1'  # fallback
    
    print(f"Using conversation template: {conv_mode}")
    print("🎤 ASR Mode - Transcribe speech to text")
    print("=" * 50)
    
    while True:
        # Get audio input
        audio_path = input("\nEnter path to audio file (or 'quit' to exit): ").strip()
        
        if audio_path.lower() in ['quit', 'exit', 'q']:
            break
            
        if not audio_path or not os.path.exists(audio_path):
            print("❌ Please provide a valid audio file path.")
            continue
            
        try:
            # Load audio
            print(f"📁 Loading audio: {audio_path}")
            target_sr = getattr(model.encoder, 'input_sampling_rate', 16000)
            # Move audio to the same device as the model
            audio, sr = load_audio_into_tensor(audio_path, target_sr=target_sr, device=args.device)
            print(f"🔊 Audio loaded: {audio.shape[0]/sr:.2f}s at {sr}Hz")
            
            # Create conversation
            conv = conv_templates[conv_mode].copy()
            
            # Create ASR prompt
            asr_prompt = f"Transcribe the following audio to text: {DEFAULT_AUDIO_TOKEN}"
            conv.append_message(conv.roles[0], asr_prompt)
            conv.append_message(conv.roles[1], None)
            
            # Get prompt
            prompt = conv.get_prompt()
            
            # Tokenize
            input_ids_list = tokenizer_mm_token(
                prompt,
                model.text_decoder.tokenizer,
                DEFAULT_AUDIO_TOKEN_IDX,
                return_tensors="pt"
            )
            # to tensor
            input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=args.device).unsqueeze(0)
            breakpoint()
            
            # Set up text streamer
            streamer = TextStreamer(
                model.text_decoder.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            
            print("\n📝 Transcription:")
            print("-" * 30)
            
            # Generate transcription
            try:
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        audios_srs=[(audio, sr)],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        streamer=streamer,
                        use_cache=True,
                    )
            except Exception as gen_e:
                # Print error message and hint about possible cause
                print(f"❌ Error processing audio: {str(gen_e)}")
                if "tuple index out of range" in str(gen_e):
                    print("⚠️  HINT: This error may be caused by a mismatch between the model and tokenizer, or by an incorrect or corrupted checkpoint. Please ensure you are using a compatible model and tokenizer, and that your checkpoint is valid.")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                continue
            
            # Get full output for logging
            try:
                outputs = model.text_decoder.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                ).strip()
            except Exception as decode_e:
                print(f"❌ Error decoding output: {str(decode_e)}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                continue
            
            if args.debug:
                print(f"\n🔍 Debug Info:")
                print(f"  Prompt: {prompt}")
                print(f"  Output: {outputs}")
                
        except Exception as e:
            print(f"❌ Error processing audio: {str(e)}")
            if "tuple index out of range" in str(e):
                print("⚠️  HINT: This error may be caused by a mismatch between the model and tokenizer, or by an incorrect or corrupted checkpoint. Please ensure you are using a compatible model and tokenizer, and that your checkpoint is valid.")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpeechLLM ASR CLI")
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to the SpeechLLM model directory"
    )
    parser.add_argument(
        "--torch-dtype", 
        type=torch_dtype_from_str, 
        default=None,
        help="PyTorch dtype for model (float32, float16, bfloat16)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run the model on"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--max-new-tokens", 
        type=int, 
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "-d", "--debug", 
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    main(args)

# Sample usage:
# python speechllm/speechllm/cli.py --model-path /path/to/speechllm/checkpoint
