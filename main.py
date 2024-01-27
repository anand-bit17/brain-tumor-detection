import roboflow
#roboflow.login()
from roboflow import Roboflow
rf = Roboflow(api_key="R8BLyvKaKnimKC2vp8wB")
project = rf.workspace().project("brain-tumor-detection-pmbvq")
model = project.version("1").model

# infer on a local image
print(model.predict("D.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("D.jpg", confidence=40, overlap=30).save("cd.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
