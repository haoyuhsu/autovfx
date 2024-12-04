# Use GPT-4 vision API to estimate the scale of an object
import os
import openai
import base64
import requests

openai.organization = "org-3RBckaRMgqYfrez6l1XnkmWi"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Available GPT models: gpt-4-vision-preview

# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

def estimate_object_scale(img_path: str = None, object_name: str = None) -> float:
    '''
    Use GPT-4 vision API to estimate the scale of an object
    Input: image path, object name
    Output: object scale (in meters)
    '''
    has_img = img_path is not None
    has_obj_name = object_name is not None

    user1 = f"I would like you to help me estimate the size of an object in real world given both its appearance and its description. The size value should be a maximum value over the height, width, length of the object. Please only give me a single estimated size value (in meters). Do not response any other texts in your estimation."
    assistant1 = f'Got it. I will complete what you give me next.'
    user2 = []
    if has_img and has_obj_name:
        user2.append({"type": "text", "text": f"What is the estimated size of this {object_name} object shown in the picture in real world? Please only give me a single estimated size value (in meters). Response with a single value. Do not response any other texts in your estimation."})
    elif has_img and not has_obj_name:
        user2.append({"type": "text", "text": f"What is the estimated size of this object shown in the picture in real world? Please only give me a single estimated size value (in meters). Response with a single value. Do not response any other texts in your estimation."})
    elif not has_img and has_obj_name:
        user2.append({"type": "text", "text": f"What is the estimated size of this {object_name} object in real world? Please only give me a single estimated size value (in meters).  Response with a single value. Do not response any other texts in your estimation."})
    else:
        raise ValueError("Either image path or object name must be provided.")
    
    if has_img:
        base64_image = encode_image(img_path)
        user2.append({
            "type": "image_url",
            "image_url": {
            "url": f"data:image/png;base64,{base64_image}"
            }
        })
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that pays attention to an object's appearance and its description, then estimates the size of the object in real world."},
        {"role": "user", "content": user1},
        {"role": "assistant", "content": assistant1},
        {"role": "user", "content": user2},
    ]

    # model_name = "gpt-4-vision-preview" if has_img else "gpt-4-1106-preview"  # `gpt-4-vision-preview` is deprecated recently
    model_name = "gpt-4o-2024-05-13"

    print('GPT model used:', model_name)

    # Option 1: use requests
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 300,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
    content = response['choices'][0]['message']['content']

    # Option 2: use openai python client
    # kwargs = {}
    # kwargs["messages"] = messages
    # kwargs["model"] = model_name
    # kwargs["max_tokens"] = 512
    # kwargs["temperature"] = 0
    # kwargs["stop"] = ["\n", " User:", " Assistant:"]
    # content = openai.chat.completions.create(**kwargs).choices[0].message.content
    
    estimated_scale = float(content)
    return estimated_scale


def estimate_object_forward_axis(img_folder: str, object_name: str) -> float:
    '''
    Use GPT-4 vision API to estimate the forward axis of an object in blender
    Input: image folder, object name
    Output: forward axis (FORWARD_X, FORWARD_Y, TRACK_NEGATIVE_X, TRACK_NEGATIVE_Y)
    '''
    user1 = f"I would like you to help me estimate which image is the frontal view of the object out of 4 possible views. Please only give me a single index value (0, 1, 2, 3) of the image that you think is the frontal view of the object. Do not response any other texts in your estimation."
    assistant1 = f'Got it. I will complete what you give me next.'
    user2 = [
        {"type": "text", "text": f"What is the index of the image that shows the frontal view of this {object_name} object? Please only give me a single index value (0, 1, 2, 3)."},
    ]

    imgs_path = sorted(os.listdir(img_folder))
    for img_path in imgs_path:
        base64_image = encode_image(os.path.join(img_folder, img_path))
        user2.append({
            "type": "image_url",
            "image_url": {
            "url": f"data:image/png;base64,{base64_image}"
            }
        })

    messages = [
        {"role": "system", "content": "You are a helpful assistant that pays attention to an object's appearance and its geometry, then estimates the frontal view of the object."},
        {"role": "user", "content": user1},
        {"role": "assistant", "content": assistant1},
        {"role": "user", "content": user2},
    ]

    model_name = "gpt-4-vision-preview"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 300,
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response = response.json()
    content = response['choices'][0]['message']['content']

    # 0: -Y, 1: +X, 2: +Y, 3: -X
    AXIS_LIST = ["TRACK_NEGATIVE_Y", "FORWARD_X", "FORWARD_Y", "TRACK_NEGATIVE_X"]
    forward_axis = AXIS_LIST[int(content)]

    return forward_axis


if __name__ == '__main__':
    # test object scale estimation
    # img_path = '../blender/images/dd324f84d894471da817ef20e3414c27/000.png'
    # object_name = 'apple'
    # estimated_scale = estimate_object_scale(img_path, object_name)
    # print(f'Estimated scale of {object_name} is {estimated_scale} meters.')

    # test object forward axis estimation
    object_name_list = ['Ford Taxi']
    root_img_folder = '../blender/images/'
    for object_name in object_name_list:
        img_folder = os.path.join(root_img_folder, object_name)
        forward_axis = estimate_object_forward_axis(img_folder, object_name)
        print(f'Estimated forward axis of {object_name} is {forward_axis}.')
