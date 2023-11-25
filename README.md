# AI-G16

To run our pipeline, please run the following commands:

1) Install dependencies

```
pip install -r requirements.txt
```

2) Download Emotion Classifier

```
gdown 1-8URvi1jyCVQIknHA5gQaEkpVWOJydha -O src/emotion_classifier/
```

3) Run Evaluation:

```
python src/evaluation.py
```

## Note

### Emotion classifier
If you want to train Emotion Classifier yourself, run:

```
python src/emotion_classifier/train.py > log.log 2>&1
```

### Elementary Discourse Units (EDUs)
If you want to run EDU segmentation, you will have to build the model for punctuation restoration from DeepPavlov:

```
https://github.com/Generative-Assistants/dream.git
```

```
cd dream
```

```
docker-compose -f docker-compose.yml -f assistant_dists/dream/docker-compose.override.yml -f assistant_dists/dream/dev.yml -f assistant_dists/dream/proxy.yml up --build sentseg
```

After that you can run ```python src/edu_segmentation/edu.py```

In this version we do not use EDU segmentation, but we will use it in competiotion itself. 

### GAT
If you want to train GAT by yourself, run 
```
python src/training.py
```

