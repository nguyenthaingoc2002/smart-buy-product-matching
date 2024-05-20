import np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

product_matching_labels_data = np.load('product_matching_labels.npz')
product_matching_labels = product_matching_labels_data["product_matching_labels"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/products/{product_id}")
async def get_list_product_id_similar(product_id: int):
  list_product_id_similar = []
  for i, value in enumerate(product_matching_labels[product_id]):
    if value == 1:
      list_product_id_similar.append(i)
  return {
    "list_product_id_similar": list_product_id_similar,
    "message": "Find list product id similar success",
    "code": 200
  }
