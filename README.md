# AI-G16

Install dependencies
```
pip install -r requirements.txt
```

## Emotion classifier

### Train

```
gdown 1-8URvi1jyCVQIknHA5gQaEkpVWOJydha -O src/emotion_classifier/
```

```
python src/emotion_classifier/train.py > log.log 2>&1
```


## Elementary Discourse Units (EDUs)

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
