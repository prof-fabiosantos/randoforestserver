import pickle
import numpy as np
import litserve as ls
from starlette.middleware.cors import CORSMiddleware


class RandomForestAPI(ls.LitAPI):
    def setup(self, device):
        # Carrega o modelo salvo
        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def decode_request(self, request):
        # Converte a entrada em numpy array
        x = np.asarray(request["input"])
        x = np.expand_dims(x, 0)
        return x

    def predict(self, x):
        return self.model.predict(x)

    def encode_response(self, output):
        return {"class_idx": int(output)}


if __name__ == "__main__":
    api = RandomForestAPI()
    server = ls.LitServer(api)

    # ✅ Habilita CORS
    server.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # pode trocar "*" por ["https://randoforestclient.vercel.app"]
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    server.run(port=8000)
