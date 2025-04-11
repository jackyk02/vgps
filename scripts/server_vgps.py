# python scripts/server_vgps.py --checkpoint="/root/V-GPS/v-gps" --port=3100

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import uvicorn
import jax
import jax.numpy as jnp
import io
from PIL import Image
import tensorflow as tf
import os
from jaxrl_m.agents import agents
from jaxrl_m.data.text_processing import text_processors
from jaxrl_m.vision import encoders
from flax.training import checkpoints
import yaml
import argparse
import time
import json_numpy as json

# Ensure TensorFlow doesn't use GPU
os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub"
tf.config.set_visible_devices([], 'GPU')

# Parse command line arguments
parser = argparse.ArgumentParser(description='V-GPS FastAPI Server')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the V-GPS checkpoint')
parser.add_argument('--config', type=str, default="experiments/configs/pretrained_checkpoint.yaml", 
                   help='Path to the config file')
parser.add_argument('--port', type=int, default=3100, help='Port number')
parser.add_argument('--host', type=str, default="127.0.0.1", help='Host address')
args = parser.parse_args()

app = FastAPI(title="V-GPS API", description="API for V-GPS value estimation")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model configuration
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Initialize the model
class ModelWrapper:
    def __init__(self, checkpoint_path, config_path):
        self.config = load_config(config_path)
        self.checkpoint_path = checkpoint_path
        self.initialize_model()
        
    def initialize_model(self):
        # Create encoder from config
        self.encoder_def = encoders[self.config["encoder"]](**self.config["encoder_kwargs"])
        
        # Example data for initialization
        example_actions = np.zeros((1, 7), dtype=np.float32)
        example_obs = {
            "image": np.zeros((1, 256, 256, 3), dtype=np.uint8)
        }
        example_batch = {
            "observations": example_obs,
            "goals": {
                "language": np.zeros((1, 512), dtype=np.float32),
            },
            "actions": example_actions,
        }

        # Create agent
        self.agent = agents[self.config["agent"]].create(
            rng=jax.random.PRNGKey(0),
            encoder_def=self.encoder_def,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            **self.config["agent_kwargs"],
        )
        
        # Load text processor
        self.text_processor = text_processors[self.config["text_processor"]]()
        
        # Restore checkpoint
        print(f"Loading checkpoint from {self.checkpoint_path}")
        self.agent = checkpoints.restore_checkpoint(self.checkpoint_path, self.agent)
        print("Model loaded successfully")
    
    def process_instruction(self, instruction):
        return self.text_processor.encode(instruction)
    
    def get_values(self, observations, goals, actions):
        return self.agent.get_q_values(observations, goals, actions)

# Initialize model
model = ModelWrapper(checkpoint_path=args.checkpoint, config_path=args.config)
print("Model initialized and ready for inference")

@app.post("/process")
async def process(request: Request):
    start_time = time.time()
    
    try:
        # Parse the request body
        body_bytes = await request.body()
        body = json.loads(body_bytes.decode())
        
        # Extract data from request
        instruction = body.get("instruction")
        image_path = body.get("image_path")
        actions = np.array(body.get("action"), dtype=np.float32)
        
        # Process instruction
        processed_instruction = model.process_instruction(instruction)
        
        # Ensure actions have correct shape
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        
        # Load and process image
        image = Image.open(image_path)
        image_array = np.array(image.resize((256, 256)), dtype=np.uint8)
        image_array = image_array[None]  # Add batch dimension
        
        # Prepare inputs
        observations = {"image": image_array}
        
        # Handle batch size matching
        num_actions = actions.shape[0]
        if num_actions > 1:
            # Duplicate the processed instruction to match the batch size of actions
            goals = {"language": np.tile(processed_instruction, (num_actions, 1))}
        else:
            goals = {"language": processed_instruction[None]}
        
        # Get values
        values = model.get_values(observations, goals, actions)
        values_list = values.tolist()
        
        processing_time = time.time() - start_time
        
        return {
            "rewards": values_list,
            "processing_time": processing_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)