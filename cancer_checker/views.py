from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from time import sleep
from django.shortcuts import render
from .Model import model0
import numpy as np
import json
import base64
from PIL import Image
from io import BytesIO
import torch

  



@csrf_exempt
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def check(request):
    classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    if request.method == 'POST':
    
        base64_image = json.loads(request.body.decode('utf-8'))['image']

        # Extract the base64 data part from the string
        image_data = base64_image.split(',')[1]

        # Decode the base64 data
        decoded_image_data = base64.b64decode(image_data)

        # Open the image using PIL
        image = Image.open(BytesIO(decoded_image_data))
        image=torch.tensor(np.asarray(image.resize((64,64))))
        image=image/255
        print(image,image.shape)
        # ypreds=model0(image.unsqueeze(dim=0).permute(0,3,1,2))
        # print(classes[torch.argmax(ypreds)])
        
        return JsonResponse({'result':'classes[torch.argmax(ypreds)]'})
       
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=405)
