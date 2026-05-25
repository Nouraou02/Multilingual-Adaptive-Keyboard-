import torch
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from MultilingualKeyboardEngine import MultilingualKeyboardEngine
from Router_class import Router
# Import your existing class and setup here
# from your_file import MultilingualKeyboardEngine 

app = FastAPI()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
local_model_path = "models/Qwen/Qwen3-1.7B" # Path to the downloaded model directory
device = torch.device("cuda:2")


# Load model in 4-bit to save memory (important for keyboard research)
Qwen_model = AutoModelForCausalLM.from_pretrained(
    local_model_path, 
    #device_map={ " " : 1},  # Automatically distribute layers across available devices
    torch_dtype=torch.float16,
    #load_in_4bit=True,
    local_files_only=True,
    trust_remote_code=True,
).to(device)

Qwen_model_router = AutoModelForCausalLM.from_pretrained(
    local_model_path, 
    #device_map={ " " : 1},  # Automatically distribute layers across available devices
    torch_dtype=torch.float16,
    #load_in_4bit=True,
    local_files_only=True,
    trust_remote_code=True,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    local_model_path, 
    local_files_only=True,
    trust_remote_code=True)

Qwen_model = PeftModel.from_pretrained(
    Qwen_model,
    "./best_adapter_enfr",
    adapter_name="fr_adapter",
    is_trainable=False
)

Qwen_model.load_adapter(
    "./zh_en_adapter_best",
    adapter_name="zh_adapter"
)

router = Router(hidden_size=2048).to(device)
router.load_state_dict(torch.load(f="./router_best.pth"))
router.to(device)

#Qwen_model.set_adapter("fr_adapter")  # or zh, just to initialize
#Qwen_model.disable_adapter_layers()   # THIS is the correct API

# 1. Initialize your model ONCE when the server starts
# Using the LoRA adapters and Qwen-1.7B base model you trained
engine = MultilingualKeyboardEngine(Qwen_model, Qwen_model_router, tokenizer, router, device)

class KeyboardRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: KeyboardRequest):
    predictions, adapter, confidence = engine.get_next_words(request.text)
    return {
        "predictions": predictions, 
        "adapter": adapter,
        "confidence": confidence
    }