import pickle
import asyncio
import np


from motor.motor_asyncio import AsyncIOMotorClient
from sklearn.metrics.pairwise import cosine_similarity

from feature_extraction import feature_extraction_name_product
from feature_extraction import feature_extraction_image_product
from math import comb
client = AsyncIOMotorClient("mongodb+srv://nguyenthaingoc162002:nguyenthaingoc123@cluster0.l9dc6na.mongodb.net/?retryWrites=true&w=majority")

database = client.test

products_collection = database.get_collection("products")

with open('src/model/product_matching_model.pkl', 'rb') as f:
    clf = pickle.load(f)


def product_helper(product) -> dict:
    return {
        "_id": str(product["_id"]),
        "id": product["id"],
        "name": product["name"],
        "e_commerce": product["e_commerce"],
        "url_product": product["url_product"],
        "url_thumbnail": product["url_thumbnail"],
        "price": product["price"],
        "description": product["description"],
        "group_id": product["group_id"]
    }

async def retrieve_products():
    products = []
    async for product in products_collection.find():
        products.append(product_helper(product))
    return products

async def main():
    products = await retrieve_products()
    number_product = len(products)
    product_names = [product["name"] for product in products]
    product_image_urls = [product["url_thumbnail"] for product in products]
    product_prices = [product["price"] for product in products]
    product__name_embeddings = feature_extraction_name_product(product_names)
    product__image_embeddings = feature_extraction_image_product(product_image_urls)
    cosine_similarity_product_names = cosine_similarity(product__name_embeddings)
    cosine_similarity_product_images = cosine_similarity(product__image_embeddings)
    product_matching_labels = np.full((number_product, number_product), -1)
    number_right = 0
    for i in range(number_product):
        for j in range(number_product):
            if product_matching_labels[i][j] == -1 and i != j:
                label =  clf.predict([[cosine_similarity_product_names[i][j],
                                            cosine_similarity_product_images[i][j],
                                            abs(product_prices[i] - product_prices[j])]])[0]
                product_matching_labels[i][j] = label
                product_matching_labels[j][i] = label
                if (product_matching_labels[i][j] == 1 and products[i]['group_id'] == products[j]['group_id']) or (product_matching_labels[i][j] == 0 and products[i]['group_id'] != products[j]['group_id']):
                    number_right += 1
    print("total product pair", comb(number_product, 2))
    print("right product pair", number_right)
    np.savez('api/product_matching_labels.npz', product_matching_labels = product_matching_labels)

if __name__ == "__main__":
    asyncio.run(main())
