from rest_framework.response import Response
from rest_framework.views import APIView
from model import utilities
import io
import base64
from PIL import Image

# Create your views here.

class detectionAPI(APIView):
    def get(self , request):
        return Response( {'msg': 'ok'},status= 200)

    def post(self , request):
        #getting image from request params
        img_data = request.FILES['image'].read()
        image = utilities.bytes_to_image(img_data)
        # model and running inference
        model = utilities.get_model()
        inference = model(image)
        inference.render()
        # Encoding results and send it as response
        buffered =  io.BytesIO()
        base64_img = Image.fromarray(inference.imgs[0])
        base64_img.save(buffered, format="JPEG")
        base64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return Response(base64_img)
