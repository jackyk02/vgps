import requests
import numpy as np
import argparse
import base64
from PIL import Image
import io
import json
import time

def main():
    parser = argparse.ArgumentParser(description='Test V-GPS API')
    parser.add_argument('--url', type=str, default='http://localhost:8000/get_values', help='API endpoint URL')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--instruction', type=str, default='put eggplant into yellow basket', help='Task instruction')
    args = parser.parse_args()
    
    # Sample actions
    actions = [
        [-3.72366211e-02, 7.17199291e-03, -1.45544186e-01, 3.92003991e-02, 7.66136944e-02, 4.05342411e-03, 8.99420261e-01],
        [-2.00744509e-02, 6.26228866e-03, -7.30025396e-02, 5.30035943e-02, 3.47160921e-02, 5.64907417e-02, 9.99990000e-01]
    ]
    
    # Load and prepare the image
    try:
        img = Image.open(args.image)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        
        # Prepare the request
        files = {
            'image': ('image.jpg', img_byte_arr.getvalue(), 'image/jpg'),
        }
        
        data = {
            'instruction': args.instruction,
            'actions': str(actions)
        }
        
        # Send the request
        start_time = time.time()
        response = requests.post(args.url, files=files, data=data)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"Request successful. Time: {request_time:.4f}s, API processing time: {result['processing_time']:.4f}s")
            print(f"Values: {result['values']}")
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()