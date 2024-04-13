sudo docker build -t flask_app -f flask_app.dockerfile .

git checkout train
cp -r ./models/bird_classification_model.h5 ./models


docker run -p 8000:8000 flask_app

