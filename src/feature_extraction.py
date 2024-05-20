from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES=True
import re
import urllib.request
import time
import np

def load_stopwords():
    stop_words = []
    with open("src/vietnamese-stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        stop_words.append(line.replace("\n",""))
    return stop_words

def preprocess_text(sentence, tokenize, stop_words):
    sentence = str(sentence).lower()
    sentence = re.sub(r"[^\w\s]", " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = tokenize(sentence)
    filtered_words = [word for word in sentence.split() if word not in stop_words]
    return ' '.join(filtered_words)

def feature_extraction_name_product(sentences):
    stop_words = load_stopwords()
    model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
    sentences = [preprocess_text(sentence, tokenize, stop_words) for sentence in sentences]
    embeddings = model.encode(sentences)
    return embeddings

def download_image(product_image_urls):
  for index, product_image_url in enumerate(product_image_urls):
    retry = 0
    while(True):
      try:
        urllib.request.urlretrieve(product_image_url, "data\images\{}.png".format(index))
        print("{} success".format(index))
        break
      except Exception as e:
        retry += 1
        time.sleep(0.5)
        print("The error is: ", e)
        print(index, product_image_url)
        if retry == 3:
          break

def extract_image_to_vector(image_urls, model):
  image_embeddings = []
  for index in range(len(image_urls)):
    image = Image.open("data/images/{index_image}.png".format(index_image = index))
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.asarray(image)
    image = preprocess_input(np.expand_dims(image.copy(), axis=0))

    resnet_feature = model.predict(image)
    resnet_feature_np = np.array(resnet_feature)
    image_embeddings.append(resnet_feature_np.flatten())

  return image_embeddings

def feature_extraction_image_product(image_urls):
#   download_image(image_urls)
  resnet = ResNet50(include_top=False, pooling='avg', weights='imagenet')
  model = Sequential()
  model.add(resnet)
  model.layers[0].trainable = False
  image_embeddings = extract_image_to_vector(image_urls, model)
  return image_embeddings
