from roboflow import Roboflow

rf = Roboflow(api_key="UkRjaQuvk8seDH3UM5io")
workspace = rf.workspace("scan-api-orl")  # Remplacez par votre workspace
project = workspace.project("shuttlecock-cqzy3-gpbq9")  # Remplacez par votre project ID
version = project.version("1")  # Remplacez par le numéro de version souhaité
model = version.model

model.download()

# model.predict("../output/frame_009674.jpg", confidence=40, overlap=30).save("prediction.jpg")
