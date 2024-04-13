## Download
docker build -t kaggle-downloader .

docker run -v $(pwd)/data:/data -e KAGGLE_USERNAME=$KAGGLE_USERNAME -e KAGGLE_KEY=$KAGGLE_KEY kaggle-downloader

git checkout -b train
cp -r ../data ./data

## Trainp
docker build -t bird-classification .
docker run bird-classification
